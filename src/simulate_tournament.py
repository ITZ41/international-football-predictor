"""
Tournament simulator using Monte Carlo simulation.
Accepts a group stage CSV, simulates the full tournament N times.
Outputs probability of each team reaching each stage.

Usage:
    python simulate_tournament.py --groups groups.csv --n_sims 10000 --output tournament_sim_results.csv

groups.csv format:
    team,group
    Brazil,A
    Argentina,A
    France,B
    ...
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
from collections import defaultdict

from features import (
    load_data, build_name_resolver, standardize_results, encode_tournament,
    get_feature_columns, H2H_WINDOW, ELO_INITIAL,
)

# -- prediction engine (same logic as predict.py) --

def get_elo(team, date, elo_history):
    history = elo_history.get(team, [])
    if not history:
        return ELO_INITIAL
    lo, hi = 0, len(history) - 1
    result = ELO_INITIAL
    while lo <= hi:
        mid = (lo + hi) // 2
        if history[mid][0] < date:
            result = history[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return result

def dc_tau(h, a, lh, la, rho):
    if h == 0 and a == 0: return 1.0 - lh * la * rho
    elif h == 1 and a == 0: return 1.0 + lh * rho
    elif h == 0 and a == 1: return 1.0 + la * rho
    elif h == 1 and a == 1: return 1.0 - rho
    else: return 1.0

def get_match_probs(home_team, away_team, artifacts, results, match_date=None):
    """
    Get win/draw/loss probabilities for a single match using the ensemble model.
    Returns (p_home, p_draw, p_away).
    """
    if match_date is None:
        match_date = pd.Timestamp.now()

    name_map = artifacts["name_map"]
    elo_history = artifacts["elo_history"]
    feature_cols = artifacts["feature_cols"]
    poisson_fc = artifacts["poisson_feature_cols"]
    poisson_scaler = artifacts["poisson_scaler"]
    rho = artifacts["dc_rho"]
    neutral = True  # tournament matches at neutral venues

    home_r = name_map.get(home_team, home_team)
    away_r = name_map.get(away_team, away_team)
    home_elo = get_elo(home_r, match_date, elo_history)
    away_elo = get_elo(away_r, match_date, elo_history)

    def form(team, w):
        h = results[(results["home_team"] == team) & (results["date"] < match_date)]
        a = results[(results["away_team"] == team) & (results["date"] < match_date)]
        hr = h[["date", "home_score", "away_score"]].copy()
        hr.columns = ["date", "gs", "gc"]
        ar = a[["date", "away_score", "home_score"]].copy()
        ar.columns = ["date", "gs", "gc"]
        all_m = pd.concat([hr, ar]).sort_values("date").tail(w)
        if len(all_m) == 0: return 0.5, 1.0, 1.0
        wr = (all_m["gs"] > all_m["gc"]).astype(float)
        wr.loc[all_m["gs"] == all_m["gc"]] = 0.5
        return wr.mean(), all_m["gs"].mean(), all_m["gc"].mean()

    hf5 = form(home_r, 5); af5 = form(away_r, 5)

    def form_gc(team, w):
        h = results[(results["home_team"] == team) & (results["date"] < match_date)]
        a = results[(results["away_team"] == team) & (results["date"] < match_date)]
        hr = h[["date", "home_score", "away_score"]].copy()
        hr.columns = ["date", "gs", "gc"]
        ar = a[["date", "away_score", "home_score"]].copy()
        ar.columns = ["date", "gs", "gc"]
        all_m = pd.concat([hr, ar]).sort_values("date").tail(w)
        if len(all_m) == 0: return 1.0, 1.0
        return all_m["gs"].mean(), all_m["gc"].mean()

    h_gs, h_gc = form_gc(home_r, 5)
    a_gs, a_gc = form_gc(away_r, 5)

    # XGB features
    xgb_fd = {
        "home_elo": home_elo, "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_advantage": 0,  # neutral
        "home_win_rate_5": hf5[0], "home_gs_avg_5": hf5[1], "home_gc_avg_5": hf5[2],
        "away_win_rate_5": af5[0], "away_gs_avg_5": af5[1], "away_gc_avg_5": af5[2],
        "home_win_rate_10": hf5[0], "home_gs_avg_10": hf5[1], "home_gc_avg_10": hf5[2],
        "away_win_rate_10": af5[0], "away_gs_avg_10": af5[1], "away_gc_avg_10": af5[2],
        "h2h_home_win_rate": 0.33, "h2h_draw_rate": 0.33, "h2h_avg_goals": 2.5,
        "home_rest_days": 5, "away_rest_days": 5,
        "home_gd_trend_5": 0.0, "away_gd_trend_5": 0.0,
        "home_comp_ratio_10": 0.8, "away_comp_ratio_10": 0.8,
        "home_win_rate_home_10": hf5[0], "home_win_rate_away_10": hf5[0],
        "away_win_rate_home_10": af5[0], "away_win_rate_away_10": af5[0],
        "home_streak": 0, "away_streak": 0,
        "home_clean_sheet_rate_10": 0.3, "away_clean_sheet_rate_10": 0.3,
    }
    tourney_type = "world_cup"
    for col in feature_cols:
        if col.startswith("tourney_"):
            xgb_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
    X_xgb = np.array([[xgb_fd.get(col, 0) for col in feature_cols]])
    xgb_proba = artifacts["model"].predict_proba(X_xgb)[0]

    # Poisson+DC
    pois_fd = {
        "elo_diff": home_elo - away_elo, "home_advantage": 0,
        "home_gs_avg_5": h_gs, "home_gc_avg_5": h_gc,
        "away_gs_avg_5": a_gs, "away_gc_avg_5": a_gc,
    }
    for col in poisson_fc:
        if col.startswith("tourney_"):
            pois_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
    X_pois = np.clip(poisson_scaler.transform(
        np.array([[pois_fd.get(col, 0) for col in poisson_fc]], dtype=np.float64)), -5, 5)
    lh = max(float(artifacts["poisson_home"].predict(X_pois)[0]), 0.01)
    la = max(float(artifacts["poisson_away"].predict(X_pois)[0]), 0.01)

    _, _, p_hw, p_d, p_aw = _poisson_dc_scorelines(lh, la, rho)

    # Ensemble (weighted average)
    w = artifacts["ensemble_weights"]
    ens_proba = w[0] * xgb_proba + w[1] * np.array([p_hw, p_d, p_aw])

    return ens_proba[0], ens_proba[1], ens_proba[2]


def _poisson_dc_scorelines(lh, la, rho, max_goals=6):
    p_hw = p_d = p_aw = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_indep = poisson.pmf(h, lh) * poisson.pmf(a, la)
            prob = p_indep * dc_tau(h, a, lh, la, rho)
            if h > a: p_hw += prob
            elif h == a: p_d += prob
            else: p_aw += prob
    return None, None, p_hw, p_d, p_aw


def precompute_match_probs(teams, artifacts, results, match_date):
    """
    Pre-compute (p_home, p_draw, p_away) for all unique unordered team pairs.
    Returns dict: (team1, team2) -> (p_home, p_draw, p_away) where team1 is 'home'.
    """
    print("  Pre-computing match probabilities...")
    prob_table = {}
    n = len(teams)
    total_pairs = n * (n - 1) // 2
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = teams[i], teams[j]
            p_home, p_draw, p_away = get_match_probs(t1, t2, artifacts, results, match_date)
            prob_table[(t1, t2)] = (p_home, p_draw, p_away)
            # Reverse: swap home/away
            prob_table[(t2, t1)] = (p_away, p_draw, p_home)
            count += 1
            if count % 20 == 0:
                print(f"    {count}/{total_pairs} pairs computed...")
    print(f"  Done. {len(prob_table)} directed pairs cached.")
    return prob_table


def simulate_group_match_from_table(team1, team2, prob_table):
    """Simulate a group match using pre-computed prob table. Returns (points1, points2)."""
    p_home, p_draw, p_away = prob_table[(team1, team2)]
    outcome = np.random.choice([0, 1, 2], p=[p_home, p_draw, p_away])
    if outcome == 0:
        return 3, 0
    elif outcome == 1:
        return 1, 1
    else:
        return 0, 3


def rank_group(teams_pts):
    """
    Rank teams by points. Tiebreak: goal difference (random here), then random.
    Returns ordered list of teams.
    """
    # Add small random tiebreaker
    teams = list(teams_pts.keys())
    pts = np.array([teams_pts[t] for t in teams], dtype=float)
    # Add tiny random noise for tiebreaking
    noise = np.random.uniform(0, 0.001, len(teams))
    order = np.argsort(-(pts + noise))
    return [teams[i] for i in order]


def simulate_knockout_match_from_table(team1, team2, prob_table):
    """Simulate a knockout match using pre-computed prob table. Returns winner."""
    p_home, p_draw, p_away = prob_table[(team1, team2)]
    # Draw -> extra time/penalties: split 50/50
    p_home_total = p_home + p_draw * 0.5
    p_away_total = p_away + p_draw * 0.5
    total = p_home_total + p_away_total
    p_home_total /= total
    return team1 if np.random.random() < p_home_total else team2


def simulate_tournament(groups, artifacts, results, n_sims=10000):
    """
    Simulate full tournament.
    groups: dict mapping group letter -> list of 4 teams
    Returns dict: team -> {stage: count}
    """
    all_teams = sorted({t for g in groups.values() for t in g})
    stages = ["Group", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
    results_count = {t: {s: 0 for s in stages} for t in all_teams}
    match_date = pd.Timestamp("2026-07-01")

    # -- Pre-compute all pairwise probabilities once --
    prob_table = precompute_match_probs(all_teams, artifacts, results, match_date)

    for sim in range(n_sims):
        if (sim + 1) % 2000 == 0:
            print(f"  Simulation {sim+1}/{n_sims}...")

        # --- Group stage ---
        group_standings = {}
        for group_name, teams in groups.items():
            pts = {t: 0 for t in teams}
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    p1, p2 = simulate_group_match_from_table(teams[i], teams[j], prob_table)
                    pts[teams[i]] += p1
                    pts[teams[j]] += p2
            ranked = rank_group(pts)
            group_standings[group_name] = ranked
            # Top 2 advance from each group
            for t in ranked[:2]:
                results_count[t]["Group"] += 1

        # --- Round of 16 pairings (standard World Cup format) ---
        group_names = sorted(groups.keys())
        if len(group_names) >= 8:
            r16_pairs = [
                (group_standings[group_names[0]][0], group_standings[group_names[1]][1]),
                (group_standings[group_names[2]][0], group_standings[group_names[3]][1]),
                (group_standings[group_names[4]][0], group_standings[group_names[5]][1]),
                (group_standings[group_names[6]][0], group_standings[group_names[7]][1]),
                (group_standings[group_names[1]][0], group_standings[group_names[0]][1]),
                (group_standings[group_names[3]][0], group_standings[group_names[2]][1]),
                (group_standings[group_names[5]][0], group_standings[group_names[4]][1]),
                (group_standings[group_names[7]][0], group_standings[group_names[6]][1]),
            ]
        elif len(group_names) >= 4:
            r16_pairs = [
                (group_standings["A"][0], group_standings["B"][1]),
                (group_standings["C"][0], group_standings["D"][1]),
                (group_standings["B"][0], group_standings["A"][1]),
                (group_standings["D"][0], group_standings["C"][1]),
            ]
            if len(group_names) >= 6:
                r16_pairs += [
                    (group_standings["E"][0], group_standings["F"][1]),
                    (group_standings["F"][0], group_standings["E"][1]),
                ]
        else:
            advancing = []
            for gn in group_names:
                advancing.extend(group_standings[gn][:2])
            r16_pairs = [(advancing[i], advancing[i+1]) for i in range(0, len(advancing)-1, 2)]

        # --- Knockout stages (all use pre-computed prob table) ---
        # Count all teams that reached Round of 16
        for t1, t2 in r16_pairs:
            results_count[t1]["Round of 16"] += 1
            results_count[t2]["Round of 16"] += 1

        r16_winners = []
        for t1, t2 in r16_pairs:
            w = simulate_knockout_match_from_table(t1, t2, prob_table)
            r16_winners.append(w)

        # Stage names based on bracket size
        n = len(r16_winners)
        if n >= 8:
            ko_stages = ["Quarter-final", "Semi-final", "Final"]
        elif n >= 4:
            ko_stages = ["Quarter-final", "Final"]
        elif n >= 2:
            ko_stages = ["Final"]
        else:
            ko_stages = []

        current = r16_winners
        for stage_name in ko_stages:
            if len(current) < 2:
                break
            # Record all teams in this stage before simulating
            for t in current:
                results_count[t][stage_name] += 1
            next_round = []
            for i in range(0, len(current) - 1, 2):
                w = simulate_knockout_match_from_table(current[i], current[i+1], prob_table)
                next_round.append(w)
            current = next_round

        if len(current) >= 1:
            results_count[current[0]]["Winner"] += 1

    # Convert counts to probabilities
    prob_results = []
    for team in sorted(all_teams):
        row = {"team": team}
        for stage in stages:
            row[stage] = round(results_count[team][stage] / n_sims * 100, 2)
        prob_results.append(row)

    return pd.DataFrame(prob_results)


def main():
    parser = argparse.ArgumentParser(description="Tournament Monte Carlo Simulator")
    parser.add_argument("--groups", required=True, help="CSV with columns: team, group")
    parser.add_argument("--n_sims", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--output", default="tournament_sim_results.csv")
    args = parser.parse_args()

    print("Loading model and data...")
    with open("models/model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    results, _, _, _ = load_data("data")
    results = standardize_results(results, artifacts["name_map"])
    results = results.sort_values("date").reset_index(drop=True)

    # Load groups
    groups_df = pd.read_csv(args.groups)
    if "team" not in groups_df.columns or "group" not in groups_df.columns:
        print("ERROR: groups.csv must have 'team' and 'group' columns")
        sys.exit(1)

    groups = {}
    for _, row in groups_df.iterrows():
        g = row["group"].strip().upper()
        if g not in groups:
            groups[g] = []
        groups[g].append(row["team"])

    print(f"\nTournament setup:")
    for g, teams in sorted(groups.items()):
        print(f"  Group {g}: {', '.join(teams)}")
    print(f"  Simulations: {args.n_sims}")

    import time
    t0 = time.time()
    print(f"\nRunning Monte Carlo simulation...")
    sim_results = simulate_tournament(groups, artifacts, results, args.n_sims)
    elapsed = time.time() - t0

    # Save results
    sim_results.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    print(f"Total time: {elapsed:.1f}s ({args.n_sims} sims)")

    # -- Formatted terminal table --
    stages = ["Group", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
    display_cols = ["team"] + stages

    # Build group lookup for display
    team_group = {}
    for g, teams in groups.items():
        for t in teams:
            team_group[t] = g

    print("\n" + "=" * 90)
    print("TOURNAMENT SIMULATION RESULTS")
    print("=" * 90)

    # Print header
    header = f"{'Team':<18} {'Grp':<4}"
    for s in stages:
        header += f" {s:>12}"
    print(header)
    print("-" * 90)

    # Sort by winner probability descending
    sorted_results = sim_results.sort_values("Winner", ascending=False)
    for _, row in sorted_results.iterrows():
        line = f"{row['team']:<18} {team_group.get(row['team'],'?'):<4}"
        for s in stages:
            line += f" {row[s]:>11.1f}%"
        print(line)

    # Top contenders
    print(f"\nTop 5 by win probability:")
    top5 = sorted_results.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {row['team']:<18} Winner: {row['Winner']:.1f}%  "
              f"Final: {row['Final']:.1f}%  SF: {row['Semi-final']:.1f}%")

    # -- Win probability bar chart --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        GROUP_COLORS = {
            "A": "#e41a1c", "B": "#377eb8", "C": "#4daf4a", "D": "#984ea3",
            "E": "#ff7f00", "F": "#ffff33", "G": "#a65628", "H": "#f781bf",
        }

        chart_data = sorted_results.head(16).sort_values("Winner", ascending=True)
        colors = [GROUP_COLORS.get(team_group.get(t, ""), "#999999") for t in chart_data["team"]]

        fig, ax = plt.subplots(figsize=(12, max(6, len(chart_data) * 0.45)))
        bars = ax.barh(chart_data["team"], chart_data["Winner"], color=colors, edgecolor="white")
        ax.set_xlabel("Win Probability (%)")
        ax.set_title("Tournament Win Probability by Team")
        for bar, val in zip(bars, chart_data["Winner"]):
            if val > 0.5:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=9)

        # Group legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=f"Group {g}")
                           for g in sorted(groups.keys())]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        chart_path = args.output.replace(".csv", "_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved bar chart to {chart_path}")
    except Exception as e:
        print(f"\n  (Chart generation skipped: {e})")

    # -- Most likely bracket path --
    print("\n" + "-" * 60)
    print("MOST LIKELY PATH (by win probability)")
    print("-" * 60)
    sorted_winners = sorted_results.iloc[0]
    print(f"  Predicted Winner: {sorted_winners['team']} ({sorted_winners['Winner']:.1f}%)")
    print(f"  Most likely finalist pool (Final >= 10%):")
    for _, row in sorted_results[sorted_results["Final"] >= 10].head(4).iterrows():
        print(f"    {row['team']:<18} Final: {row['Final']:.1f}%  Winner: {row['Winner']:.1f}%")
    print()


if __name__ == "__main__":
    main()
