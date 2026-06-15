"""
Prediction CLI for international football matches.
Three models: XGBoost, Poisson+Dixon-Coles, Ensemble.

Single:
    python predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral False
Batch:
    python predict.py --batch fixtures.csv --output predictions.csv
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson

from features import (
    get_feature_columns, H2H_WINDOW,
    ELO_INITIAL, ELO_K, ELO_HOME_BOOST,
    encode_tournament, is_competitive_tournament,
    load_data, build_name_resolver, standardize_results,
)

TARGET_NAMES = ["Home Win", "Draw", "Away Win"]
MAX_GOALS = 6


def load_artifacts(path="model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run train.py first.")
        sys.exit(1)


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


def poisson_dc_scorelines(lh, la, rho, max_goals=MAX_GOALS):
    score_probs = {}
    best_prob = 0
    best_score = (0, 0)
    p_hw = p_d = p_aw = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_indep = poisson.pmf(h, lh) * poisson.pmf(a, la)
            prob = p_indep * dc_tau(h, a, lh, la, rho)
            score_probs[(h, a)] = prob
            if prob > best_prob:
                best_prob = prob
                best_score = (h, a)
            if h > a: p_hw += prob
            elif h == a: p_d += prob
            else: p_aw += prob
    return best_score, score_probs, p_hw, p_d, p_aw


def _team_matches_before(results, team, date):
    """Get all matches for a team before a given date, with gs, gc, tournament columns."""
    h = results[(results["home_team"] == team) & (results["date"] < date)]
    a = results[(results["away_team"] == team) & (results["date"] < date)]
    hr = h[["date", "home_score", "away_score", "tournament"]].copy()
    hr.columns = ["date", "gs", "gc", "tournament"]
    ar = a[["date", "away_score", "home_score", "tournament"]].copy()
    ar.columns = ["date", "gs", "gc", "tournament"]
    return pd.concat([hr, ar]).sort_values("date")


def _team_home_matches_before(results, team, date):
    """Get home-only matches for a team before a given date."""
    h = results[(results["home_team"] == team) & (results["date"] < date)]
    return h[["date", "home_score", "away_score"]].copy().rename(
        columns={"home_score": "gs", "away_score": "gc"}).sort_values("date")


def _team_away_matches_before(results, team, date):
    """Get away-only matches for a team before a given date."""
    a = results[(results["away_team"] == team) & (results["date"] < date)]
    return a[["date", "away_score", "home_score"]].copy().rename(
        columns={"away_score": "gs", "home_score": "gc"}).sort_values("date")


def compute_all_features_for_match(home_team, away_team, tournament, neutral, match_date,
                                     results, elo_history, name_map):
    """
    Compute ALL features for a single match using only data before match_date.
    Mirrors build_feature_matrix() logic for prediction time — no hardcoded defaults.
    Returns (xgb_fd, pois_fd, home_elo, away_elo, h2h, tourney_type).
    """
    home_r = name_map.get(home_team, home_team)
    away_r = name_map.get(away_team, away_team)
    home_elo = get_elo(home_r, match_date, elo_history)
    away_elo = get_elo(away_r, match_date, elo_history)

    # --- Form features (windows 5 and 10) ---
    def form(team, w):
        m = _team_matches_before(results, team, match_date).tail(w)
        if len(m) == 0:
            return 0.5, 1.0, 1.0
        wr = (m["gs"] > m["gc"]).astype(float)
        wr.loc[m["gs"] == m["gc"]] = 0.5
        return wr.mean(), m["gs"].mean(), m["gc"].mean()

    hf5 = form(home_r, 5)
    hf10 = form(home_r, 10)
    af5 = form(away_r, 5)
    af10 = form(away_r, 10)

    # --- Goals scored/conceded (window 5, for Poisson) ---
    def form_gc(team, w):
        m = _team_matches_before(results, team, match_date).tail(w)
        if len(m) == 0:
            return 1.0, 1.0
        return m["gs"].mean(), m["gc"].mean()

    h_gs, h_gc = form_gc(home_r, 5)
    a_gs, a_gc = form_gc(away_r, 5)

    # --- H2H ---
    pair_mask = (
        ((results["home_team"] == home_r) & (results["away_team"] == away_r)) |
        ((results["home_team"] == away_r) & (results["away_team"] == home_r))
    )
    h2h_all = results[pair_mask]
    h2h = h2h_all[h2h_all["date"] < match_date]
    recent_h2h = h2h.tail(H2H_WINDOW)
    if len(recent_h2h) > 0:
        hw = d = tg = 0
        for _, m in recent_h2h.iterrows():
            tg += m["home_score"] + m["away_score"]
            if m["home_score"] == m["away_score"]: d += 1
            if (m["home_team"] == home_r and m["home_score"] > m["away_score"]) or \
               (m["away_team"] == home_r and m["away_score"] > m["home_score"]):
                hw += 1
        n = len(recent_h2h)
        h2h_hwr, h2h_dr, h2h_ag = hw / n, d / n, tg / n
    else:
        h2h_hwr, h2h_dr, h2h_ag = 0.33, 0.33, 2.5

    # --- Rest days ---
    def rest(team):
        dates = pd.concat([
            results.loc[(results["home_team"] == team) & (results["date"] < match_date), "date"],
            results.loc[(results["away_team"] == team) & (results["date"] < match_date), "date"]
        ]).sort_values()
        return (match_date - dates.iloc[-1]).days if len(dates) > 0 else 30

    # --- GD trend (window=5) ---
    def gd_trend(team, w=5):
        m = _team_matches_before(results, team, match_date).tail(w)
        if len(m) == 0:
            return 0.0
        return (m["gs"] - m["gc"]).mean()

    # --- Competitive ratio (window=10) ---
    def comp_ratio(team, w=10):
        m = _team_matches_before(results, team, match_date).tail(w)
        if len(m) == 0:
            return 0.5
        return m["tournament"].apply(is_competitive_tournament).mean()

    # --- Home/away split form (window=10) ---
    def home_wr_at_home(team, w=10):
        m = _team_home_matches_before(results, team, match_date).tail(w)
        if len(m) == 0: return 0.5
        wr = (m["gs"] > m["gc"]).astype(float)
        wr.loc[m["gs"] == m["gc"]] = 0.5
        return wr.mean()

    def away_wr(team, w=10):
        m = _team_away_matches_before(results, team, match_date).tail(w)
        if len(m) == 0: return 0.5
        wr = (m["gs"] > m["gc"]).astype(float)
        wr.loc[m["gs"] == m["gc"]] = 0.5
        return wr.mean()

    # --- Streak ---
    def calc_streak(team):
        m = _team_matches_before(results, team, match_date)
        if len(m) == 0:
            return 0
        results_list = (m["gs"] > m["gc"]).astype(float)
        results_list.loc[m["gs"] == m["gc"]] = 0.5
        vals = results_list.values
        if len(vals) == 0:
            return 0
        last = vals[-1]
        if last == 0.5:
            return 0
        streak = 1 if last == 1.0 else -1
        for i in range(len(vals) - 2, -1, -1):
            if vals[i] == last and vals[i] != 0.5:
                streak += 1 if vals[i] == 1.0 else -1
            else:
                break
        return streak

    # --- Clean sheet rate (window=10) ---
    def clean_sheet_rate(team, w=10):
        m = _team_matches_before(results, team, match_date).tail(w)
        if len(m) == 0:
            return 0.3
        return (m["gc"] == 0).mean()

    # --- Last-3 scoring ---
    def last3_gs(team):
        m = _team_matches_before(results, team, match_date).tail(3)
        if len(m) == 0:
            return 1.0
        return m["gs"].mean()

    # --- Elo percentile ---
    def elo_pct(team):
        team_elo = get_elo(team, match_date, elo_history)
        # Build distribution from elo_history final values
        all_final = []
        for t, hist in elo_history.items():
            if hist:
                # Get Elo at match_date
                all_final.append(get_elo(t, match_date, elo_history))
        all_sorted = np.sort(all_final)
        return np.searchsorted(all_sorted, team_elo) / max(len(all_sorted) - 1, 1)

    tourney_type = encode_tournament(tournament)

    h_home_wr = home_wr_at_home(home_r)
    h_away_wr = away_wr(home_r)
    a_home_wr = home_wr_at_home(away_r)
    a_away_wr = away_wr(away_r)

    xgb_fd = {
        "home_elo": home_elo, "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_advantage": 0 if neutral else 1,
        "home_win_rate_5": hf5[0], "home_gs_avg_5": hf5[1], "home_gc_avg_5": hf5[2],
        "away_win_rate_5": af5[0], "away_gs_avg_5": af5[1], "away_gc_avg_5": af5[2],
        "home_win_rate_10": hf10[0], "home_gs_avg_10": hf10[1], "home_gc_avg_10": hf10[2],
        "away_win_rate_10": af10[0], "away_gs_avg_10": af10[1], "away_gc_avg_10": af10[2],
        "h2h_home_win_rate": h2h_hwr, "h2h_draw_rate": h2h_dr, "h2h_avg_goals": h2h_ag,
        "home_rest_days": rest(home_r), "away_rest_days": rest(away_r),
        "home_gd_trend_5": gd_trend(home_r), "away_gd_trend_5": gd_trend(away_r),
        "home_comp_ratio_10": comp_ratio(home_r), "away_comp_ratio_10": comp_ratio(away_r),
        "home_win_rate_home_10": h_home_wr, "home_win_rate_away_10": h_away_wr,
        "away_win_rate_home_10": a_home_wr, "away_win_rate_away_10": a_away_wr,
        "home_streak": calc_streak(home_r), "away_streak": calc_streak(away_r),
        "home_clean_sheet_rate_10": clean_sheet_rate(home_r),
        "away_clean_sheet_rate_10": clean_sheet_rate(away_r),
        "home_last3_gs_avg": last3_gs(home_r), "away_last3_gs_avg": last3_gs(away_r),
        "home_elo_pct": elo_pct(home_r), "away_elo_pct": elo_pct(away_r),
        "elo_pct_diff": elo_pct(home_r) - elo_pct(away_r),
    }

    pois_fd = {
        "elo_diff": home_elo - away_elo,
        "home_advantage": 0 if neutral else 1,
        "home_gs_avg_5": h_gs, "home_gc_avg_5": h_gc,
        "away_gs_avg_5": a_gs, "away_gc_avg_5": a_gc,
    }

    return xgb_fd, pois_fd, home_elo, away_elo, h2h_all, tourney_type


def predict_single(home_team, away_team, tournament, neutral, artifacts, results):
    """Run all three models on a single match."""
    match_date = pd.Timestamp.now()
    name_map = artifacts["name_map"]
    elo_history = artifacts["elo_history"]
    feature_cols = artifacts["feature_cols"]
    poisson_fc = artifacts["poisson_feature_cols"]
    poisson_scaler = artifacts["poisson_scaler"]
    rho = artifacts["dc_rho"]

    xgb_fd, pois_fd, home_elo, away_elo, h2h, tourney_type = compute_all_features_for_match(
        home_team, away_team, tournament, neutral, match_date, results, elo_history, name_map
    )

    # XGBoost
    for col in feature_cols:
        if col.startswith("tourney_"):
            xgb_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
    X_xgb = np.array([[xgb_fd.get(col, 0) for col in feature_cols]])
    xgb_proba = artifacts["model"].predict_proba(X_xgb)[0]
    thresholds = artifacts["optimal_thresholds"]
    above = [c for c in range(3) if xgb_proba[c] > thresholds[c]]
    xgb_pred = max(above, key=lambda c: xgb_proba[c]) if above else np.argmax(xgb_proba)

    # Poisson + Dixon-Coles
    for col in poisson_fc:
        if col.startswith("tourney_"):
            pois_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
    X_pois_raw = np.array([[pois_fd.get(col, 0) for col in poisson_fc]], dtype=np.float64)
    X_pois = np.clip(poisson_scaler.transform(X_pois_raw), -5, 5)
    lh = max(float(artifacts["poisson_home"].predict(X_pois)[0]), 0.01)
    la = max(float(artifacts["poisson_away"].predict(X_pois)[0]), 0.01)
    best_score, score_probs, p_hw, p_d, p_aw = poisson_dc_scorelines(lh, la, rho)
    top_scores = sorted(score_probs.items(), key=lambda x: -x[1])[:5]

    # Ensemble
    meta_input = np.hstack([xgb_proba, [p_hw, p_d, p_aw]]).reshape(1, -1)
    ensemble_proba = artifacts["meta_model"].predict_proba(meta_input)[0]

    # H2H
    home_r = name_map.get(home_team, home_team)
    away_r = name_map.get(away_team, away_team)
    pair_mask = (
        ((results["home_team"] == home_r) & (results["away_team"] == away_r)) |
        ((results["home_team"] == away_r) & (results["away_team"] == home_r))
    )
    h2h_matches = results[pair_mask]
    recent = h2h_matches.tail(10)
    h2h_hw = h2h_d = h2h_aw = 0
    h2h_list = []
    for _, m in recent.iterrows():
        if m["home_score"] > m["away_score"]:
            if m["home_team"] == home_r: h2h_hw += 1
            else: h2h_aw += 1
        elif m["home_score"] < m["away_score"]:
            if m["away_team"] == away_r: h2h_aw += 1
            else: h2h_hw += 1
        else:
            h2h_d += 1
        h2h_list.append(m)

    # Output
    nl = "=" * 55
    print(f"\n{nl}")
    print(f"  {home_team} vs {away_team}")
    print(f"  {tournament} | {'Neutral venue' if neutral else 'Home venue'}")
    print(nl)

    print(f"\n  XGBoost model:")
    print(f"  {home_team} Win: {xgb_proba[0]*100:5.1f}%  |  "
          f"Draw: {xgb_proba[1]*100:5.1f}%  |  "
          f"{away_team} Win: {xgb_proba[2]*100:5.1f}%")

    print(f"\n  Poisson+DC model:")
    print(f"  {home_team} Win: {p_hw*100:5.1f}%  |  "
          f"Draw: {p_d*100:5.1f}%  |  "
          f"{away_team} Win: {p_aw*100:5.1f}%")

    print(f"\n  Ensemble model:")
    print(f"  {home_team} Win: {ensemble_proba[0]*100:5.1f}%  |  "
          f"Draw: {ensemble_proba[1]*100:5.1f}%  |  "
          f"{away_team} Win: {ensemble_proba[2]*100:5.1f}%")

    print(f"\n  Expected goals: {home_team} {lh:.2f} - {la:.2f} {away_team}")

    print(f"\n  Top scorelines:")
    for (h, a), p in top_scores:
        print(f"    {h}-{a}: {p*100:.1f}%")

    print(f"\n  Elo: {home_team} {home_elo:.0f} | {away_team} {away_elo:.0f}")

    if len(recent) > 0:
        print(f"\n  H2H last {len(recent)}: {home_team} {h2h_hw}W {h2h_d}D {h2h_aw}L")
        for m in reversed(h2h_list[-5:]):
            print(f"    {m['date'].strftime('%Y-%m-%d')}: "
                  f"{m['home_team']} {int(m['home_score'])}-{int(m['away_score'])} {m['away_team']}")

    print(f"\n{nl}")

    return {
        "home_team": home_team, "away_team": away_team,
        "xgb_hw": xgb_proba[0], "xgb_d": xgb_proba[1], "xgb_aw": xgb_proba[2],
        "pois_hw": p_hw, "pois_d": p_d, "pois_aw": p_aw,
        "ens_hw": ensemble_proba[0], "ens_d": ensemble_proba[1], "ens_aw": ensemble_proba[2],
        "exp_hg": lh, "exp_ag": la,
        "elo_home": home_elo, "elo_away": away_elo,
        "best_scoreline": f"{best_score[0]}-{best_score[1]}",
    }


def predict_batch(fixtures_csv, output_csv, artifacts, results):
    """Batch prediction with all three models."""
    fixtures = pd.read_csv(fixtures_csv)
    if "tournament" not in fixtures.columns:
        fixtures["tournament"] = "Friendly"
    if "neutral" not in fixtures.columns:
        fixtures["neutral"] = False

    predictions = []
    for _, row in fixtures.iterrows():
        neutral = str(row.get("neutral", "False")).lower() in ("true", "1", "yes")
        tournament = str(row.get("tournament", "Friendly"))
        match_date = pd.Timestamp.now()
        name_map = artifacts["name_map"]
        elo_history = artifacts["elo_history"]
        feature_cols = artifacts["feature_cols"]
        poisson_fc = artifacts["poisson_feature_cols"]
        poisson_scaler = artifacts["poisson_scaler"]
        rho = artifacts["dc_rho"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        try:
            xgb_fd, pois_fd, home_elo, away_elo, _, tourney_type = compute_all_features_for_match(
                home_team, away_team, tournament, neutral, match_date, results, elo_history, name_map
            )

            for col in feature_cols:
                if col.startswith("tourney_"):
                    xgb_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
            X_xgb = np.array([[xgb_fd.get(col, 0) for col in feature_cols]])
            xgb_proba = artifacts["model"].predict_proba(X_xgb)[0]
            thresholds = artifacts["optimal_thresholds"]
            above = [c for c in range(3) if xgb_proba[c] > thresholds[c]]
            xgb_pred = max(above, key=lambda c: xgb_proba[c]) if above else np.argmax(xgb_proba)

            for col in poisson_fc:
                if col.startswith("tourney_"):
                    pois_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
            X_pois_raw = np.array([[pois_fd.get(col, 0) for col in poisson_fc]], dtype=np.float64)
            X_pois = np.clip(poisson_scaler.transform(X_pois_raw), -5, 5)
            lh = max(float(artifacts["poisson_home"].predict(X_pois)[0]), 0.01)
            la = max(float(artifacts["poisson_away"].predict(X_pois)[0]), 0.01)
            best_sl, _, p_hw, p_d, p_aw = poisson_dc_scorelines(lh, la, rho)

            meta_input = np.hstack([xgb_proba, [p_hw, p_d, p_aw]]).reshape(1, -1)
            ens_proba = artifacts["meta_model"].predict_proba(meta_input)[0]

            predictions.append({
                "home_team": home_team, "away_team": away_team,
                "tournament": tournament, "neutral": neutral,
                "xgb_pred": TARGET_NAMES[xgb_pred],
                "xgb_hw": round(xgb_proba[0], 4), "xgb_d": round(xgb_proba[1], 4),
                "xgb_aw": round(xgb_proba[2], 4),
                "pois_hw": round(p_hw, 4), "pois_d": round(p_d, 4),
                "pois_aw": round(p_aw, 4),
                "ens_hw": round(ens_proba[0], 4), "ens_d": round(ens_proba[1], 4),
                "ens_aw": round(ens_proba[2], 4),
                "exp_hg": round(lh, 2), "exp_ag": round(la, 2),
                "elo_home": round(home_elo, 0), "elo_away": round(away_elo, 0),
                "best_scoreline": f"{best_sl[0]}-{best_sl[1]}",
            })
        except Exception as e:
            print(f"  ERROR {home_team} vs {away_team}: {e}")
            predictions.append({"home_team": home_team, "away_team": away_team, "error": str(e)})

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(predictions)} predictions to {output_csv}")
    for _, p in df.iterrows():
        if "error" not in p:
            print(f"  {p['home_team']} vs {p['away_team']}: "
                  f"XGB={p['xgb_pred']} ({p['xgb_hw']*100:.0f}/{p['xgb_d']*100:.0f}/{p['xgb_aw']*100:.0f}%) | "
                  f"Ens=({p['ens_hw']*100:.0f}/{p['ens_d']*100:.0f}/{p['ens_aw']*100:.0f}%) | "
                  f"Score: {p['best_scoreline']}")


def main():
    parser = argparse.ArgumentParser(description="Predict international football matches")
    parser.add_argument("--home", help="Home team name")
    parser.add_argument("--away", help="Away team name")
    parser.add_argument("--tournament", default="Friendly")
    parser.add_argument("--neutral", default="False")
    parser.add_argument("--batch", help="CSV fixtures file")
    parser.add_argument("--output", default="batch_predictions.csv")
    args = parser.parse_args()

    artifacts = load_artifacts()
    results, _, _, _ = load_data(".")
    results = standardize_results(results, artifacts["name_map"])
    results = results.sort_values("date").reset_index(drop=True)

    if args.batch:
        predict_batch(args.batch, args.output, artifacts, results)
    else:
        if not args.home or not args.away:
            parser.error("--home and --away required")
        neutral = args.neutral.lower() in ("true", "1", "yes")
        predict_single(args.home, args.away, args.tournament, neutral, artifacts, results)


if __name__ == "__main__":
    main()
