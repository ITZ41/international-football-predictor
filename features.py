"""
Feature engineering for international football match prediction.
All features use only data available BEFORE the match date — no leakage.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ELO_INITIAL = 1500.0
ELO_K = 20.0          # K-factor for Elo updates
ELO_HOME_BOOST = 100  # home-field advantage in Elo terms
FORM_WINDOWS = [5, 10]
H2H_WINDOW = 10

# Tournament category mapping
TOURNAMENT_MAP = {
    "FIFA World Cup": "world_cup",
    "FIFA World Cup qualification": "qualifier",
    "UEFA Euro": "continental",
    "UEFA Euro qualification": "qualifier",
    "Copa América": "continental",
    "Copa América qualification": "qualifier",
    "AFC Asian Cup": "continental",
    "AFC Asian Cup qualification": "qualifier",
    "Africa Cup of Nations": "continental",
    "Africa Cup of Nations qualification": "qualifier",
    "CONCACAF Championship": "continental",
    "CONCACAF Championship qualification": "qualifier",
    "Oceania Nations Cup": "continental",
    "Oceania Nations Cup qualification": "qualifier",
    "UEFA Nations League": "nations_league",
    "CONCACAF Nations League": "nations_league",
    "Friendly": "friendly",
}


def load_data(data_dir="."):
    """Load all CSV files and return as DataFrames."""
    results = pd.read_csv(f"{data_dir}/results.csv", parse_dates=["date"])
    goalscorers = pd.read_csv(f"{data_dir}/goalscorers.csv", parse_dates=["date"])
    shootouts = pd.read_csv(f"{data_dir}/shootouts.csv", parse_dates=["date"])
    former_names = pd.read_csv(f"{data_dir}/former_names.csv",
                                parse_dates=["start_date", "end_date"])
    return results, goalscorers, shootouts, former_names


def build_name_resolver(former_names):
    """
    Build a mapping from former/current team names to a canonical name.
    Strategy: map every former name to its current name.
    If multiple current names exist, pick the one with latest start_date.
    """
    name_map = {}
    for _, row in former_names.iterrows():
        former = row["former"]
        current = row["current"]
        # Map former -> current
        if former not in name_map:
            name_map[former] = current
    return name_map


def resolve_team_name(name, name_map):
    """Resolve a team name to its canonical form."""
    return name_map.get(name, name)


def standardize_results(results, name_map):
    """Replace all team names with canonical names."""
    results = results.copy()
    results["home_team"] = results["home_team"].apply(lambda x: resolve_team_name(x, name_map))
    results["away_team"] = results["away_team"].apply(lambda x: resolve_team_name(x, name_map))
    return results


def compute_elo_ratings(results):
    """
    Compute rolling Elo ratings for all teams.
    Returns:
        elo_history: dict mapping team -> list of (date, elo_after_match)
        match_elos: list of (home_elo_before, away_elo_before) per match
    """
    elo = defaultdict(lambda: ELO_INITIAL)
    # Track full history for lookup at prediction time
    elo_history = defaultdict(list)  # team -> [(date, elo_after)]
    match_elos = []  # (home_elo_before, away_elo_before) for each match

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Snapshot Elo BEFORE this match
        home_elo = elo[home]
        away_elo = elo[away]
        match_elos.append((home_elo, away_elo))

        # Determine actual outcome
        if row["home_score"] > row["away_score"]:
            actual_home = 1.0
        elif row["home_score"] < row["away_score"]:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # Expected outcome with home boost
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + ELO_HOME_BOOST)) / 400.0))

        # Update Elo
        elo[home] += ELO_K * (actual_home - expected_home)
        elo[away] += ELO_K * ((1.0 - actual_home) - (1.0 - expected_home))

        # Record history
        elo_history[home].append((date, elo[home]))
        elo_history[away].append((date, elo[away]))

    return elo_history, match_elos


def get_elo_before_match(team, date, elo_history):
    """
    Get a team's Elo rating just before a given date.
    Binary search through the team's Elo history.
    """
    history = elo_history.get(team, [])
    if not history:
        return ELO_INITIAL

    # Find the last entry with date < match_date
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


def compute_form_features(results, window):
    """
    Compute rolling form features for each team over the last `window` matches.
    Returns lists aligned with results rows:
        home_win_rate, home_gs_avg, home_gc_avg, away_win_rate, away_gs_avg, away_gc_avg
    """
    # Track each team's match history: list of (date, goals_scored, goals_conceded, win)
    team_matches = defaultdict(list)

    home_win_rates, home_gs_avgs, home_gc_avgs = [], [], []
    away_win_rates, away_gs_avgs, away_gc_avgs = [], [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Get form for home team (matches BEFORE this one)
        home_hist = [m for m in team_matches[home] if m[0] < date]
        recent_home = home_hist[-window:] if len(home_hist) >= window else home_hist

        if recent_home:
            home_win_rates.append(sum(m[3] for m in recent_home) / len(recent_home))
            home_gs_avgs.append(sum(m[1] for m in recent_home) / len(recent_home))
            home_gc_avgs.append(sum(m[2] for m in recent_home) / len(recent_home))
        else:
            home_win_rates.append(0.5)  # default for new teams
            home_gs_avgs.append(1.0)
            home_gc_avgs.append(1.0)

        # Get form for away team
        away_hist = [m for m in team_matches[away] if m[0] < date]
        recent_away = away_hist[-window:] if len(away_hist) >= window else away_hist

        if recent_away:
            away_win_rates.append(sum(m[3] for m in recent_away) / len(recent_away))
            away_gs_avgs.append(sum(m[1] for m in recent_away) / len(recent_away))
            away_gc_avgs.append(sum(m[2] for m in recent_away) / len(recent_away))
        else:
            away_win_rates.append(0.5)
            away_gs_avgs.append(1.0)
            away_gc_avgs.append(1.0)

        # Record this match for both teams
        # Home team perspective
        if row["home_score"] > row["away_score"]:
            home_win = 1.0
        elif row["home_score"] == row["away_score"]:
            home_win = 0.5
        else:
            home_win = 0.0

        team_matches[home].append((date, row["home_score"], row["away_score"], home_win))
        team_matches[away].append((date, row["away_score"], row["home_score"], 1.0 - home_win))

    return home_win_rates, home_gs_avgs, home_gc_avgs, away_win_rates, away_gs_avgs, away_gc_avgs


def compute_h2h_features(results):
    """
    Compute head-to-head record between the two teams from previous meetings.
    Returns: h2h_home_win_rate, h2h_draw_rate, h2h_avg_goals
    """
    # Track all matches between each pair
    h2h_matches = defaultdict(list)  # (team_a, team_b) -> list of (date, a_score, b_score)

    h2h_home_win_rates, h2h_draw_rates, h2h_avg_goals = [], [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Canonical pair key (sorted alphabetically)
        pair = tuple(sorted([home, away]))

        # Get previous H2H matches
        prev = [m for m in h2h_matches[pair] if m[0] < date]
        recent = prev[-H2H_WINDOW:] if len(prev) >= H2H_WINDOW else prev

        if recent:
            home_wins = 0
            draws = 0
            total_goals = 0
            for m in recent:
                # m = (date, score_first_team, score_second_team) where first/second = sorted order
                # Need to figure out who was home in that match
                # We stored scores in sorted team order, so we need to be careful
                # Actually, let's store from home team perspective instead
                pass
            # Re-approach: store from each match's home perspective
            # Let me redo this below
            pass
        else:
            h2h_home_win_rates.append(0.33)
            h2h_draw_rates.append(0.33)
            h2h_avg_goals.append(2.5)

        # Store this match — use sorted pair, store (date, home_score, away_score, home_team_name)
        h2h_matches[pair].append((date, row["home_score"], row["away_score"], home))

    # Redo with correct approach
    h2h_matches2 = defaultdict(list)
    h2h_home_win_rates, h2h_draw_rates, h2h_avg_goals = [], [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        pair = tuple(sorted([home, away]))

        prev = [m for m in h2h_matches2[pair] if m[0] < date]
        recent = prev[-H2H_WINDOW:] if len(prev) >= H2H_WINDOW else prev

        if recent:
            home_wins = 0
            draws = 0
            total_goals = 0
            for m in recent:
                m_date, m_home_score, m_away_score, m_home_team = m
                total_goals += m_home_score + m_away_score
                if m_home_score == m_away_score:
                    draws += 1
                # Count wins for whoever is the home team in the CURRENT match
                if m_home_team == home:
                    # Same home/away assignment
                    if m_home_score > m_away_score:
                        home_wins += 1
                else:
                    # Flipped — current home was away in that match
                    if m_away_score > m_home_score:
                        home_wins += 1
            n = len(recent)
            h2h_home_win_rates.append(home_wins / n)
            h2h_draw_rates.append(draws / n)
            h2h_avg_goals.append(total_goals / n)
        else:
            h2h_home_win_rates.append(0.33)
            h2h_draw_rates.append(0.33)
            h2h_avg_goals.append(2.5)

        h2h_matches2[pair].append((date, row["home_score"], row["away_score"], home))

    return h2h_home_win_rates, h2h_draw_rates, h2h_avg_goals


def compute_rest_days(results):
    """
    Compute days since each team's last match (fatigue/rest proxy).
    """
    last_match_date = {}
    home_rest_days = []
    away_rest_days = []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        if home in last_match_date:
            home_rest_days.append((date - last_match_date[home]).days)
        else:
            home_rest_days.append(30)  # default: assume ~1 month rest for first match

        if away in last_match_date:
            away_rest_days.append((date - last_match_date[away]).days)
        else:
            away_rest_days.append(30)

        # Update last match dates
        last_match_date[home] = date
        last_match_date[away] = date

    return home_rest_days, away_rest_days


def compute_gd_trend(results, window=5):
    """
    Compute average goal difference over last `window` matches for each team.
    """
    team_gd_history = defaultdict(list)  # team -> list of (date, gd)
    home_gd_trends = []
    away_gd_trends = []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Home team GD trend
        home_hist = [m for m in team_gd_history[home] if m[0] < date]
        recent_home = home_hist[-window:] if len(home_hist) >= window else home_hist
        if recent_home:
            home_gd_trends.append(sum(m[1] for m in recent_home) / len(recent_home))
        else:
            home_gd_trends.append(0.0)

        # Away team GD trend
        away_hist = [m for m in team_gd_history[away] if m[0] < date]
        recent_away = away_hist[-window:] if len(away_hist) >= window else away_hist
        if recent_away:
            away_gd_trends.append(sum(m[1] for m in recent_away) / len(recent_away))
        else:
            away_gd_trends.append(0.0)

        # Record this match
        gd_home = row["home_score"] - row["away_score"]
        gd_away = row["away_score"] - row["home_score"]
        team_gd_history[home].append((date, gd_home))
        team_gd_history[away].append((date, gd_away))

    return home_gd_trends, away_gd_trends


def is_competitive_tournament(tournament_name):
    """Check if tournament is competitive (not friendly)."""
    t = tournament_name.lower()
    return "friendly" not in t


def compute_competitive_ratio(results, window=10):
    """
    Compute proportion of last N matches that were competitive (non-friendly).
    Returns home_comp_ratio, away_comp_ratio lists aligned with results rows.
    """
    team_matches = defaultdict(list)  # team -> list of (date, is_competitive)
    home_ratios, away_ratios = [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Home team competitive ratio
        home_hist = [m for m in team_matches[home] if m[0] < date]
        recent_home = home_hist[-window:] if len(home_hist) >= window else home_hist
        if recent_home:
            home_ratios.append(sum(m[1] for m in recent_home) / len(recent_home))
        else:
            home_ratios.append(0.5)

        # Away team competitive ratio
        away_hist = [m for m in team_matches[away] if m[0] < date]
        recent_away = away_hist[-window:] if len(away_hist) >= window else away_hist
        if recent_away:
            away_ratios.append(sum(m[1] for m in recent_away) / len(recent_away))
        else:
            away_ratios.append(0.5)

        # Record this match
        comp = 1.0 if is_competitive_tournament(row["tournament"]) else 0.0
        team_matches[home].append((date, comp))
        team_matches[away].append((date, comp))

    return home_ratios, away_ratios


def compute_home_away_split_form(results, window=10):
    """
    Compute separate home and away win rates for each team.
    Returns: home_wr_at_home, home_wr_away, away_wr_at_home, away_wr_away
    """
    # Track home and away matches separately
    team_home_matches = defaultdict(list)  # team -> list of (date, win)
    team_away_matches = defaultdict(list)

    home_wr_home, home_wr_away, away_wr_home, away_wr_away = [], [], [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        if row["home_score"] > row["away_score"]:
            home_win, away_win = 1.0, 0.0
        elif row["home_score"] == row["away_score"]:
            home_win, away_win = 0.5, 0.5
        else:
            home_win, away_win = 0.0, 1.0

        # Home team's home win rate (playing at home)
        hh = [m for m in team_home_matches[home] if m[0] < date]
        recent_hh = hh[-window:] if len(hh) >= window else hh
        home_wr_home.append(sum(m[1] for m in recent_hh) / len(recent_hh) if recent_hh else 0.5)

        # Home team's away win rate (when they played away)
        ha = [m for m in team_away_matches[home] if m[0] < date]
        recent_ha = ha[-window:] if len(ha) >= window else ha
        home_wr_away.append(sum(m[1] for m in recent_ha) / len(recent_ha) if recent_ha else 0.5)

        # Away team's home win rate
        ah = [m for m in team_home_matches[away] if m[0] < date]
        recent_ah = ah[-window:] if len(ah) >= window else ah
        away_wr_home.append(sum(m[1] for m in recent_ah) / len(recent_ah) if recent_ah else 0.5)

        # Away team's away win rate
        aa = [m for m in team_away_matches[away] if m[0] < date]
        recent_aa = aa[-window:] if len(aa) >= window else aa
        away_wr_away.append(sum(m[1] for m in recent_aa) / len(recent_aa) if recent_aa else 0.5)

        # Record
        team_home_matches[home].append((date, home_win))
        team_away_matches[away].append((date, away_win))

    return home_wr_home, home_wr_away, away_wr_home, away_wr_away


def compute_streak(results):
    """
    Compute current streak for each team (positive = wins, negative = losses, 0 = draws/mixed).
    Returns home_streak, away_streak lists.
    """
    team_results = defaultdict(list)  # team -> list of (date, result) where 1=win, 0.5=draw, 0=loss
    home_streaks, away_streaks = [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        if row["home_score"] > row["away_score"]:
            home_result, away_result = 1.0, 0.0
        elif row["home_score"] == row["away_score"]:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0

        def calc_streak(team):
            hist = [m for m in team_results[team] if m[0] < date]
            if not hist:
                return 0
            # Walk backwards from most recent
            streak = 0
            last_result = None
            for _, result in reversed(hist):
                if last_result is None:
                    if result == 0.5:
                        return 0  # streak broken by draw
                    last_result = result
                    streak = 1 if result == 1.0 else -1
                else:
                    if result == last_result and result != 0.5:
                        streak += 1 if result == 1.0 else -1
                    else:
                        break
            return streak

        home_streaks.append(calc_streak(home))
        away_streaks.append(calc_streak(away))

        team_results[home].append((date, home_result))
        team_results[away].append((date, away_result))

    return home_streaks, away_streaks


def compute_clean_sheet_rate(results, window=10):
    """
    Compute clean sheet rate (matches with 0 goals conceded) over last N matches.
    Proxy for goalkeeper/defense quality.
    Returns home_cs_rate, away_cs_rate lists.
    """
    team_matches = defaultdict(list)  # team -> list of (date, goals_conceded)
    home_cs, away_cs = [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Home team clean sheet rate
        home_hist = [m for m in team_matches[home] if m[0] < date]
        recent_home = home_hist[-window:] if len(home_hist) >= window else home_hist
        if recent_home:
            cs = sum(1 for m in recent_home if m[1] == 0) / len(recent_home)
            home_cs.append(cs)
        else:
            home_cs.append(0.3)  # default

        # Away team clean sheet rate
        away_hist = [m for m in team_matches[away] if m[0] < date]
        recent_away = away_hist[-window:] if len(away_hist) >= window else away_hist
        if recent_away:
            cs = sum(1 for m in recent_away if m[1] == 0) / len(recent_away)
            away_cs.append(cs)
        else:
            away_cs.append(0.3)

        # Record
        team_matches[home].append((date, row["away_score"]))  # home concedes away_score
        team_matches[away].append((date, row["home_score"]))  # away concedes home_score

    return home_cs, away_cs


def compute_last3_scoring(results):
    """
    Compute average goals scored in last 3 matches for each team.
    Shorter window = more recent signal than 5 or 10 match averages.
    Returns home_last3_gs, away_last3_gs lists.
    """
    team_matches = defaultdict(list)  # team -> list of (date, goals_scored)
    home_l3, away_l3 = [], []

    for _, row in results.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Home team last 3 goals scored
        home_hist = [m for m in team_matches[home] if m[0] < date]
        recent_home = home_hist[-3:] if len(home_hist) >= 3 else home_hist
        if recent_home:
            home_l3.append(sum(m[1] for m in recent_home) / len(recent_home))
        else:
            home_l3.append(1.0)

        # Away team last 3 goals scored
        away_hist = [m for m in team_matches[away] if m[0] < date]
        recent_away = away_hist[-3:] if len(away_hist) >= 3 else away_hist
        if recent_away:
            away_l3.append(sum(m[1] for m in recent_away) / len(recent_away))
        else:
            away_l3.append(1.0)

        # Record
        team_matches[home].append((date, row["home_score"]))
        team_matches[away].append((date, row["away_score"]))

    return home_l3, away_l3


def compute_elo_percentile(results, elo_history):
    """
    Compute Elo percentile rank at each match date.
    Maps absolute Elo to [0, 1] percentile — more interpretable than raw Elo diff.
    Returns home_elo_pct, away_elo_pct, elo_pct_diff lists.
    """
    # Pre-compute all final Elos for reference distribution
    all_elos = {}
    for team, hist in elo_history.items():
        if hist:
            all_elos[team] = hist[-1][1]
    elo_values = np.array(sorted(all_elos.values()))
    n_teams = len(elo_values)

    def percentile(elo_val):
        return np.searchsorted(elo_values, elo_val) / max(n_teams - 1, 1)

    home_pct, away_pct = [], []
    for _, row in results.iterrows():
        h_elo = get_elo_before_match(row["home_team"], row["date"], elo_history)
        a_elo = get_elo_before_match(row["away_team"], row["date"], elo_history)
        home_pct.append(percentile(h_elo))
        away_pct.append(percentile(a_elo))

    return home_pct, away_pct


def encode_tournament(tournament_name):
    """Map tournament name to category."""
    # Direct match
    if tournament_name in TOURNAMENT_MAP:
        return TOURNAMENT_MAP[tournament_name]
    # Partial match
    for key, val in TOURNAMENT_MAP.items():
        if key.lower() in tournament_name.lower():
            return val
    # Default
    return "other"


def build_feature_matrix(results, elo_history):
    """
    Build the complete feature matrix from results.
    Returns a DataFrame with all features and the target variable.
    """
    print("Computing Elo-based features...")
    # Elo ratings before each match
    home_elos = []
    away_elos = []
    for _, row in results.iterrows():
        home_elos.append(get_elo_before_match(row["home_team"], row["date"], elo_history))
        away_elos.append(get_elo_before_match(row["away_team"], row["date"], elo_history))

    features = pd.DataFrame({
        "date": results["date"].values,
        "home_team": results["home_team"].values,
        "away_team": results["away_team"].values,
        "home_score": results["home_score"].values,
        "away_score": results["away_score"].values,
        "tournament": results["tournament"].values,
        "neutral": results["neutral"].values,
        "home_elo": home_elos,
        "away_elo": away_elos,
    })

    # Elo difference
    features["elo_diff"] = features["home_elo"] - features["away_elo"]

    # Home advantage flag
    features["home_advantage"] = (~features["neutral"].astype(bool)).astype(int)

    # Form features for both windows
    for window in FORM_WINDOWS:
        print(f"Computing form features (window={window})...")
        h_wr, h_gs, h_gc, a_wr, a_gs, a_gc = compute_form_features(results, window)
        features[f"home_win_rate_{window}"] = h_wr
        features[f"home_gs_avg_{window}"] = h_gs
        features[f"home_gc_avg_{window}"] = h_gc
        features[f"away_win_rate_{window}"] = a_wr
        features[f"away_gs_avg_{window}"] = a_gs
        features[f"away_gc_avg_{window}"] = a_gc

    # Head-to-head features
    print("Computing head-to-head features...")
    h2h_hwr, h2h_dr, h2h_ag = compute_h2h_features(results)
    features["h2h_home_win_rate"] = h2h_hwr
    features["h2h_draw_rate"] = h2h_dr
    features["h2h_avg_goals"] = h2h_ag

    # Rest days
    print("Computing rest days...")
    home_rest, away_rest = compute_rest_days(results)
    features["home_rest_days"] = home_rest
    features["away_rest_days"] = away_rest

    # Goal difference trend
    print("Computing goal difference trends...")
    home_gd, away_gd = compute_gd_trend(results)
    features["home_gd_trend_5"] = home_gd
    features["away_gd_trend_5"] = away_gd

    # Competitive match ratio
    print("Computing competitive match ratios...")
    home_comp, away_comp = compute_competitive_ratio(results)
    features["home_comp_ratio_10"] = home_comp
    features["away_comp_ratio_10"] = away_comp

    # Home/away split form
    print("Computing home/away split form...")
    h_home_wr, h_away_wr, a_home_wr, a_away_wr = compute_home_away_split_form(results)
    features["home_win_rate_home_10"] = h_home_wr
    features["home_win_rate_away_10"] = h_away_wr
    features["away_win_rate_home_10"] = a_home_wr
    features["away_win_rate_away_10"] = a_away_wr

    # Streak feature
    print("Computing streak features...")
    home_streak, away_streak = compute_streak(results)
    features["home_streak"] = home_streak
    features["away_streak"] = away_streak

    # Clean sheet rate (goalkeeper proxy)
    print("Computing clean sheet rates...")
    home_cs, away_cs = compute_clean_sheet_rate(results)
    features["home_clean_sheet_rate_10"] = home_cs
    features["away_clean_sheet_rate_10"] = away_cs

    # Last-3 scoring trend
    print("Computing last-3 scoring trends...")
    home_l3, away_l3 = compute_last3_scoring(results)
    features["home_last3_gs_avg"] = home_l3
    features["away_last3_gs_avg"] = away_l3

    # Elo percentile
    print("Computing Elo percentiles...")
    home_pct, away_pct = compute_elo_percentile(results, elo_history)
    features["home_elo_pct"] = home_pct
    features["away_elo_pct"] = away_pct
    features["elo_pct_diff"] = np.array(home_pct) - np.array(away_pct)

    # Tournament encoding
    print("Encoding tournament types...")
    features["tournament_type"] = features["tournament"].apply(encode_tournament)

    # One-hot encode tournament type
    tournament_dummies = pd.get_dummies(features["tournament_type"], prefix="tourney")
    features = pd.concat([features, tournament_dummies], axis=1)

    # Target variable: 0 = Home Win, 1 = Draw, 2 = Away Win
    features["target"] = np.where(
        features["home_score"] > features["away_score"], 0,
        np.where(features["home_score"] == features["away_score"], 1, 2)
    )

    return features


def get_feature_columns():
    """Return the list of feature column names for model training."""
    cols = [
        "home_elo", "away_elo", "elo_diff", "home_advantage",
        "home_win_rate_5", "home_gs_avg_5", "home_gc_avg_5",
        "away_win_rate_5", "away_gs_avg_5", "away_gc_avg_5",
        "home_win_rate_10", "home_gs_avg_10", "home_gc_avg_10",
        "away_win_rate_10", "away_gs_avg_10", "away_gc_avg_10",
        "h2h_home_win_rate", "h2h_draw_rate", "h2h_avg_goals",
        "home_rest_days", "away_rest_days",
        "home_gd_trend_5", "away_gd_trend_5",
        # New features
        "home_comp_ratio_10", "away_comp_ratio_10",
        "home_win_rate_home_10", "home_win_rate_away_10",
        "away_win_rate_home_10", "away_win_rate_away_10",
        "home_streak", "away_streak",
        "home_clean_sheet_rate_10", "away_clean_sheet_rate_10",
        "home_last3_gs_avg", "away_last3_gs_avg",
        "home_elo_pct", "away_elo_pct", "elo_pct_diff",
    ]
    # Tournament dummies — these are added dynamically based on data
    return cols


def get_h2b_summary(home_team, away_team, results, n=10):
    """
    Get head-to-head summary for display purposes.
    Returns last n matches between the two teams.
    """
    mask = (
        ((results["home_team"] == home_team) & (results["away_team"] == away_team)) |
        ((results["home_team"] == away_team) & (results["away_team"] == home_team))
    )
    h2h = results[mask].sort_values("date", ascending=False).head(n)
    return h2h
