"""
End-to-end test pipeline for international football prediction.
Exits with code 0 if all tests pass, 1 if any fail.

Usage:
    python test_pipeline.py
"""

import sys
import pickle
import numpy as np
import pandas as pd

errors = []
warnings_list = []


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not passed:
        errors.append(name)
    return passed


def warn(name, detail=""):
    msg = f"  [WARN] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    warnings_list.append(name)


# ================================================================== #
# TEST 1: Model artifacts
# ================================================================== #
print("\n" + "=" * 60)
print("TEST 1: Model Artifacts")
print("=" * 60)

try:
    with open("model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    report("model.pkl loads", True)
except Exception as e:
    report("model.pkl loads", False, str(e))
    print("\nFATAL: Cannot load model. Aborting.")
    sys.exit(1)

required_keys = [
    "model", "feature_cols", "poisson_home", "poisson_away",
    "poisson_feature_cols", "poisson_scaler", "dc_rho",
    "meta_model", "optimal_thresholds", "elo_history",
    "final_elos", "name_map",
]
missing = [k for k in required_keys if k not in artifacts]
report("All required keys present", len(missing) == 0,
       f"Missing: {missing}" if missing else f"{len(required_keys)} keys OK")

# Check model can predict
try:
    from features import get_feature_columns, H2H_WINDOW, ELO_INITIAL
    fc = artifacts["feature_cols"]
    dummy = np.zeros((1, len(fc)))
    proba = artifacts["model"].predict_proba(dummy)
    report("Model predicts", proba.shape == (1, 3),
           f"shape={proba.shape}")
    report("Probabilities sum to 1", abs(proba.sum() - 1.0) < 0.01,
           f"sum={proba.sum():.4f}")
except Exception as e:
    report("Model predicts", False, str(e))

# Check Poisson models
try:
    pfc = artifacts["poisson_feature_cols"]
    pois_dummy = np.zeros((1, len(pfc)))
    lh = artifacts["poisson_home"].predict(pois_dummy)[0]
    la = artifacts["poisson_away"].predict(pois_dummy)[0]
    report("Poisson models predict", lh > 0 and la > 0,
           f"lh={lh:.2f}, la={la:.2f}")
except Exception as e:
    report("Poisson models predict", False, str(e))


# ================================================================== #
# TEST 2: Predictions on 2022 WC matches
# ================================================================== #
print("\n" + "=" * 60)
print("TEST 2: 2022 World Cup Predictions (top-2 accuracy)")
print("=" * 60)

from features import load_data, standardize_results, encode_tournament
from predict import compute_all_features_for_match, dc_tau, poisson_dc_scorelines

results, _, _, _ = load_data(".")
results = standardize_results(results, artifacts["name_map"])
results = results.sort_values("date").reset_index(drop=True)

# 2022 WC matches with known outcomes
# Format: (home, away, tournament, neutral, date, actual_result)
# actual_result: 0=Home Win, 1=Draw, 2=Away Win
wc_matches = [
    ("Qatar", "Ecuador", "FIFA World Cup", True, "2022-11-20", 2),     # 0-2
    ("England", "Iran", "FIFA World Cup", True, "2022-11-21", 0),       # 6-2
    ("Argentina", "Saudi Arabia", "FIFA World Cup", True, "2022-11-22", 2),  # 1-2 upset
    ("France", "Australia", "FIFA World Cup", True, "2022-11-22", 0),   # 4-1
    ("Argentina", "Mexico", "FIFA World Cup", True, "2022-11-26", 0),   # 2-0
]

name_map = artifacts["name_map"]
elo_history = artifacts["elo_history"]
feature_cols = artifacts["feature_cols"]
poisson_fc = artifacts["poisson_feature_cols"]
poisson_scaler = artifacts["poisson_scaler"]
rho = artifacts["dc_rho"]

top2_correct = 0
top1_correct = 0

for home, away, tournament, neutral, date_str, actual in wc_matches:
    try:
        match_date = pd.Timestamp(date_str)
        xgb_fd, pois_fd, home_elo, away_elo, h2h, tourney_type = \
            compute_all_features_for_match(
                home, away, tournament, neutral, match_date,
                results, elo_history, name_map
            )

        for col in feature_cols:
            if col.startswith("tourney_"):
                xgb_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
        X_xgb = np.array([[xgb_fd.get(col, 0) for col in feature_cols]])
        xgb_proba = artifacts["model"].predict_proba(X_xgb)[0]

        meta_input = np.hstack([xgb_proba, [0.33, 0.34, 0.33]]).reshape(1, -1)
        ens_proba = artifacts["meta_model"].predict_proba(meta_input)[0]

        top2 = set(np.argsort(ens_proba)[-2:])
        is_top2 = actual in top2
        is_top1 = np.argmax(ens_proba) == actual
        if is_top2:
            top2_correct += 1
        if is_top1:
            top1_correct += 1

        result_names = ["H", "D", "A"]
        print(f"  {home:<18} vs {away:<18} | "
              f"Actual: {result_names[actual]} | "
              f"Pred: {result_names[np.argmax(ens_proba)]} "
              f"({ens_proba[0]*100:.0f}/{ens_proba[1]*100:.0f}/{ens_proba[2]*100:.0f}%) | "
              f"Top2: {'Y' if is_top2 else 'N'}")
    except Exception as e:
        print(f"  {home} vs {away}: ERROR — {e}")

pct = top2_correct / len(wc_matches) * 100
report(f"WC top-2 accuracy ({top2_correct}/{len(wc_matches)})", top2_correct >= 3,
       f"{pct:.0f}% (need >= 60%)")
print(f"    Top-1 accuracy: {top1_correct}/{len(wc_matches)} ({top1_correct/len(wc_matches)*100:.0f}%)")


# ================================================================== #
# TEST 3: Tournament simulator
# ================================================================== #
print("\n" + "=" * 60)
print("TEST 3: Tournament Simulator")
print("=" * 60)

import tempfile
import os

wc2022_groups = pd.DataFrame({
    "team": ["Qatar", "Ecuador", "Senegal", "Netherlands",
             "England", "Iran", "USA", "Wales",
             "Argentina", "Saudi Arabia", "Mexico", "Poland",
             "France", "Australia", "Denmark", "Tunisia"],
    "group": ["A", "A", "A", "A", "B", "B", "B", "B",
              "C", "C", "C", "C", "D", "D", "D", "D"],
})

try:
    from simulate_tournament import simulate_tournament

    groups = {}
    for _, row in wc2022_groups.iterrows():
        g = row["group"].strip().upper()
        groups.setdefault(g, []).append(row["team"])

    wc_results = simulate_tournament(groups, artifacts, results, n_sims=1000)
    report("Simulator runs without error", True)

    # Check Argentina's win probability (they won, should be reasonably high)
    arg_row = wc_results[wc_results["team"] == "Argentina"]
    if len(arg_row) > 0:
        arg_win = arg_row.iloc[0]["Winner"]
        report(f"Argentina win prob ({arg_win:.1f}%) in [5%, 40%]",
               5 <= arg_win <= 40,
               f"Got {arg_win:.1f}%")
    else:
        warn("Argentina not in results")

    # Check France's win probability
    fra_row = wc_results[wc_results["team"] == "France"]
    if len(fra_row) > 0:
        fra_win = fra_row.iloc[0]["Winner"]
        report(f"France win prob ({fra_win:.1f}%) in [5%, 40%]",
               5 <= fra_win <= 40,
               f"Got {fra_win:.1f}%")

    # Check Qatar (host, weak team) has low probability
    qat_row = wc_results[wc_results["team"] == "Qatar"]
    if len(qat_row) > 0:
        qat_win = qat_row.iloc[0]["Winner"]
        report(f"Qatar win prob ({qat_win:.1f}%) < 10%",
               qat_win < 10,
               f"Got {qat_win:.1f}%")

    # Verify probabilities sum reasonably
    group_sum = wc_results["Group"].sum()
    report(f"Group stage probabilities sum ~{group_sum:.0f}%",
           700 <= group_sum <= 900,
           f"4 groups x 2 advance x 100% = 800% expected")

    report("Output CSV has all stages", "Winner" in wc_results.columns and len(wc_results) == 16)

except Exception as e:
    report("Simulator runs without error", False, str(e))


# ================================================================== #
# SUMMARY
# ================================================================== #
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

total_tests = 10  # approximate
failed = len(errors)
warned = len(warnings_list)

if failed == 0:
    print(f"\n  ALL TESTS PASSED ({total_tests} checks, {warned} warnings)")
    if warnings_list:
        for w in warnings_list:
            print(f"    Warning: {w}")
    print()
    sys.exit(0)
else:
    print(f"\n  {failed} TEST(S) FAILED:")
    for e in errors:
        print(f"    FAIL: {e}")
    if warnings_list:
        print(f"  {warnings_list} warnings")
    print()
    sys.exit(1)
