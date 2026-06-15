"""
Train models to predict international football match outcomes.
1) XGBoost classifier (Home Win / Draw / Away Win) with class weights + threshold tuning
2) LightGBM classifier — same setup, keep the better of XGB/LGB
3) Isotonic calibration on the best model
4) Poisson regression models for home/away goals + Dixon-Coles correction
5) LogisticRegression ensemble blending best classifier + Poisson probabilities

Train: before 2018 | Validation: 2014-2017 | Test: 2018-2024
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (classification_report, accuracy_score, log_loss,
                              f1_score, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import (
    load_data, build_name_resolver, standardize_results,
    compute_elo_ratings, build_feature_matrix,
    get_feature_columns, ELO_INITIAL, ELO_K, ELO_HOME_BOOST
)

DATA_DIR = "."
TARGET_NAMES = ["Home Win", "Draw", "Away Win"]


# ------------------------------------------------------------------
# Dixon-Coles correction
# ------------------------------------------------------------------
def estimate_dc_rho(results_train):
    home_goals = results_train["home_score"].values.astype(float)
    away_goals = results_train["away_score"].values.astype(float)
    lh_mean = home_goals.mean()
    la_mean = away_goals.mean()

    def neg_log_likelihood(rho):
        ll = 0.0
        for i in range(len(home_goals)):
            h, a = int(home_goals[i]), int(away_goals[i])
            p_indep = poisson.pmf(h, lh_mean) * poisson.pmf(a, la_mean)
            tau = dc_tau(h, a, lh_mean, la_mean, rho)
            p_corrected = max(p_indep * tau, 1e-10)
            ll += np.log(p_corrected)
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-0.3, 0.3), method='bounded')
    rho = result.x
    print(f"  Estimated Dixon-Coles rho: {rho:.5f}")
    return rho


def dc_tau(h, a, lh, la, rho):
    if h == 0 and a == 0: return 1.0 - lh * la * rho
    elif h == 1 and a == 0: return 1.0 + lh * rho
    elif h == 0 and a == 1: return 1.0 + la * rho
    elif h == 1 and a == 1: return 1.0 - rho
    else: return 1.0


def poisson_dc_scorelines(lh, la, rho, max_goals=6):
    score_probs = {}
    best_prob = 0
    best_score = (0, 0)
    p_hw = p_d = p_aw = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_indep = poisson.pmf(h, lh) * poisson.pmf(a, la)
            tau = dc_tau(h, a, lh, la, rho)
            prob = p_indep * tau
            score_probs[(h, a)] = prob
            if prob > best_prob:
                best_prob = prob
                best_score = (h, a)
            if h > a: p_hw += prob
            elif h == a: p_d += prob
            else: p_aw += prob
    return best_score, score_probs, p_hw, p_d, p_aw


def poisson_win_draw_loss_dc(lh_arr, la_arr, rho, max_goals=6):
    n = len(lh_arr)
    probs = np.zeros((n, 3))
    for i in range(n):
        lh = max(lh_arr[i], 0.01)
        la = max(la_arr[i], 0.01)
        _, _, p_hw, p_d, p_aw = poisson_dc_scorelines(lh, la, rho, max_goals)
        probs[i] = [p_hw, p_d, p_aw]
    return probs


# ------------------------------------------------------------------
# Threshold tuning
# ------------------------------------------------------------------
def find_optimal_thresholds(y_true, y_proba, n_classes=3):
    best_f1 = -1
    best_thresholds = np.ones(n_classes) / n_classes
    for t0 in np.arange(0.1, 0.95, 0.05):
        for t1 in np.arange(0.1, 0.95, 0.05):
            for t2 in np.arange(0.1, 0.95, 0.05):
                thresholds = np.array([t0, t1, t2])
                preds = np.zeros(len(y_true), dtype=int)
                for j in range(len(y_true)):
                    above = [c for c in range(n_classes) if y_proba[j, c] > thresholds[c]]
                    preds[j] = max(above, key=lambda c: y_proba[j, c]) if above else np.argmax(y_proba[j])
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = thresholds.copy()
    print(f"  Optimal thresholds: {dict(zip(TARGET_NAMES, best_thresholds.round(3)))}")
    print(f"  Best macro F1 from threshold tuning: {best_f1:.4f}")
    return best_thresholds


# ------------------------------------------------------------------
# Poisson model training
# ------------------------------------------------------------------
def train_poisson_models(results, features):
    print("\nTraining Poisson regression models for goal scoring...")
    poisson_feature_cols = [
        "elo_diff", "home_advantage",
        "home_gs_avg_5", "home_gc_avg_5",
        "away_gs_avg_5", "away_gc_avg_5",
    ]
    tourney_cols = [c for c in features.columns if c.startswith("tourney_")]
    valid_tourney = [c for c in tourney_cols if features[c].sum() > 10]
    poisson_feature_cols.extend(valid_tourney)

    train_mask = features["date"] < "2018-01-01"
    test_mask = (features["date"] >= "2018-01-01") & (features["date"] < "2025-01-01")

    X_train_raw = features.loc[train_mask, poisson_feature_cols].values.astype(np.float64)
    y_home_train = features.loc[train_mask, "home_score"].values.astype(float)
    y_away_train = features.loc[train_mask, "away_score"].values.astype(float)
    X_test_raw = features.loc[test_mask, poisson_feature_cols].values.astype(np.float64)

    scaler = StandardScaler()
    X_train = np.clip(scaler.fit_transform(X_train_raw), -5, 5)
    X_test = np.clip(scaler.transform(X_test_raw), -5, 5)

    poisson_home = PoissonRegressor(alpha=1e-6, max_iter=10000, tol=1e-10)
    poisson_home.fit(X_train, y_home_train)
    poisson_away = PoissonRegressor(alpha=1e-6, max_iter=10000, tol=1e-10)
    poisson_away.fit(X_train, y_away_train)

    home_pred = poisson_home.predict(X_test)
    away_pred = poisson_away.predict(X_test)
    y_home_test = features.loc[test_mask, "home_score"].values.astype(float)
    y_away_test = features.loc[test_mask, "away_score"].values.astype(float)
    print(f"  Home goals - MAE: {np.mean(np.abs(home_pred - y_home_test)):.3f}")
    print(f"  Away goals - MAE: {np.mean(np.abs(away_pred - y_away_test)):.3f}")

    print("\nEstimating Dixon-Coles rho parameter...")
    train_results = results[results["date"] < "2018-01-01"].copy()
    rho = estimate_dc_rho(train_results)

    poisson_dc_probs = poisson_win_draw_loss_dc(home_pred, away_pred, rho)
    poisson_dc_preds = np.argmax(poisson_dc_probs, axis=1)
    print("\n  Poisson + Dixon-Coles Classification Report:")
    print(classification_report(y_true=features.loc[test_mask, "target"].values,
                                y_pred=poisson_dc_preds, target_names=TARGET_NAMES, digits=3))
    print(f"  Poisson+DC Accuracy: {accuracy_score(features.loc[test_mask, 'target'].values, poisson_dc_preds):.4f}")
    print(f"  Poisson+DC Log Loss: {log_loss(features.loc[test_mask, 'target'].values, poisson_dc_probs):.4f}")

    return poisson_home, poisson_away, poisson_feature_cols, scaler, rho


# ------------------------------------------------------------------
# Ensemble meta-model
# ------------------------------------------------------------------
def train_ensemble(cls_proba_val, poisson_proba_val, y_val):
    print("\nTraining ensemble meta-model...")
    X_meta = np.hstack([cls_proba_val, poisson_proba_val])
    meta_model = LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0)
    meta_model.fit(X_meta, y_val)
    meta_proba = meta_model.predict_proba(X_meta)
    meta_preds = meta_model.predict(X_meta)
    print(f"  Ensemble validation accuracy: {accuracy_score(y_val, meta_preds):.4f}")
    print(f"  Ensemble validation log-loss: {log_loss(y_val, meta_proba):.4f}")
    print(f"  Ensemble validation macro F1: {f1_score(y_val, meta_preds, average='macro'):.4f}")
    return meta_model


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Loading data...")
    results, goalscorers, shootouts, former_names = load_data(DATA_DIR)
    print(f"  Total matches: {len(results)}")

    name_map = build_name_resolver(former_names)
    results = standardize_results(results, name_map)
    results = results.sort_values("date").reset_index(drop=True)

    print("\nComputing Elo ratings...")
    elo_history, _ = compute_elo_ratings(results)
    all_teams = set(results["home_team"].unique()) | set(results["away_team"].unique())
    final_elos = {t: elo_history.get(t, [(None, ELO_INITIAL)])[-1][1] for t in all_teams}
    elo_df = pd.DataFrame([
        {"team": t, "elo": e} for t, e in sorted(final_elos.items(), key=lambda x: -x[1])
    ])
    elo_df.to_csv("elo_ratings.csv", index=False)
    print(f"  Saved elo_ratings.csv ({len(elo_df)} teams)")

    print("\nBuilding feature matrix...")
    features = build_feature_matrix(results, elo_history)
    print(f"  Features shape: {features.shape}")

    # Splits
    features = features[features["date"] >= "1950-01-01"].copy()
    train_mask = features["date"] < "2018-01-01"
    test_mask = (features["date"] >= "2018-01-01") & (features["date"] < "2025-01-01")
    thresh_train_mask = features["date"] < "2014-01-01"
    thresh_val_mask = (features["date"] >= "2014-01-01") & (features["date"] < "2018-01-01")

    train = features[train_mask].copy()
    test = features[test_mask].copy()
    print(f"\nTrain: {len(train)} | Test: {len(test)}")

    feature_cols = get_feature_columns()
    tourney_cols = [c for c in features.columns if c.startswith("tourney_")]
    feature_cols.extend(tourney_cols)
    print(f"Feature count: {len(feature_cols)}")

    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values

    # Class weights
    n_samples = len(y_train)
    class_counts = np.bincount(y_train.astype(int), minlength=3)
    sample_weights = np.ones(n_samples)
    for c in range(3):
        sample_weights[y_train == c] = n_samples / (3 * class_counts[c])
    sample_weights[y_train == 1] *= 1.5

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    print("\nTraining XGBoost...")
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [200, 400], "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1], "subsample": [0.8],
        "colsample_bytree": [0.8], "min_child_weight": [3, 5],
    }
    xgb_base = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0,
    )
    search = RandomizedSearchCV(xgb_base, param_grid, n_iter=20, cv=tscv,
                                 scoring="neg_log_loss", random_state=42, verbose=1, n_jobs=-1)
    search.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_model = search.best_estimator_
    xgb_proba_test = xgb_model.predict_proba(X_test)
    xgb_acc = accuracy_score(y_test, np.argmax(xgb_proba_test, axis=1))
    print(f"  XGBoost test accuracy: {xgb_acc:.4f}")

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    lgb_model = None
    lgb_acc = 0.0
    try:
        from lightgbm import LGBMClassifier
        print("\nTraining LightGBM...")
        lgb_param_grid = {
            "n_estimators": [200, 400], "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1], "subsample": [0.8],
            "colsample_bytree": [0.8], "min_child_samples": [20, 40],
            "reg_alpha": [0.1], "reg_lambda": [1.0],
        }
        lgb_base = LGBMClassifier(
            objective="multiclass", num_class=3, verbose=-1,
            random_state=42, n_jobs=-1,
        )
        lgb_search = RandomizedSearchCV(lgb_base, lgb_param_grid, n_iter=20, cv=tscv,
                                         scoring="neg_log_loss", random_state=42,
                                         verbose=1, n_jobs=-1)
        lgb_search.fit(X_train, y_train, sample_weight=sample_weights)
        lgb_model = lgb_search.best_estimator_
        lgb_proba_test = lgb_model.predict_proba(X_test)
        lgb_acc = accuracy_score(y_test, np.argmax(lgb_proba_test, axis=1))
        print(f"  LightGBM test accuracy: {lgb_acc:.4f}")
    except ImportError:
        print("\n  LightGBM not installed — skipping. pip install lightgbm to enable.")

    # ------------------------------------------------------------------
    # Pick best classifier
    # ------------------------------------------------------------------
    if lgb_model is not None and lgb_acc > xgb_acc:
        best_cls_model = lgb_model
        best_cls_name = "LightGBM"
        best_cls_proba_test = lgb_proba_test
        print(f"\n  >> LightGBM wins ({lgb_acc:.4f} vs {xgb_acc:.4f})")
    else:
        best_cls_model = xgb_model
        best_cls_name = "XGBoost"
        best_cls_proba_test = xgb_proba_test
        if lgb_model is not None:
            print(f"\n  >> XGBoost wins ({xgb_acc:.4f} vs {lgb_acc:.4f})")
        else:
            print(f"\n  >> Using XGBoost (LightGBM unavailable)")

    # ------------------------------------------------------------------
    # Isotonic calibration on best classifier
    # ------------------------------------------------------------------
    print(f"\nCalibrating {best_cls_name} with isotonic regression...")
    # Use cross-validation on training set for calibration
    cal_model = CalibratedClassifierCV(best_cls_model, method='isotonic', cv=5)
    cal_model.fit(X_train, y_train)
    cal_proba_test = cal_model.predict_proba(X_test)
    cal_preds = np.argmax(cal_proba_test, axis=1)
    cal_acc = accuracy_score(y_test, cal_preds)
    cal_logloss = log_loss(y_test, cal_proba_test)
    print(f"  Calibrated accuracy: {cal_acc:.4f}")
    print(f"  Calibrated log loss: {cal_logloss:.4f}")
    print(f"  Calibrated macro F1: {f1_score(y_test, cal_preds, average='macro'):.4f}")

    # Brier score (multi-class: mean of per-class Brier)
    brier_scores = []
    for i in range(3):
        y_bin = (y_test == i).astype(int)
        brier_scores.append(brier_score_loss(y_bin, cal_proba_test[:, i]))
    print(f"  Mean Brier score: {np.mean(brier_scores):.4f}")

    # ------------------------------------------------------------------
    # Two-stage prediction: draw classifier + H/A classifier
    # ------------------------------------------------------------------
    print("\nTraining two-stage draw + H/A classifiers...")

    # Stage 1: Draw vs Not-Draw binary classifier
    X_th_train = features.loc[thresh_train_mask, feature_cols].values
    y_th_train = features.loc[thresh_train_mask, "target"].values
    X_th_val = features.loc[thresh_val_mask, feature_cols].values
    y_th_val = features.loc[thresh_val_mask, "target"].values

    y_draw_train = (y_th_train == 1).astype(int)
    y_draw_val = (y_th_val == 1).astype(int)

    # Draw classifier with high draw weight
    draw_model = CalibratedClassifierCV(
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0),
        method='isotonic', cv=5
    )
    draw_sw = np.ones(len(y_draw_train))
    sw_draw_pos = len(y_draw_train) / (2 * y_draw_train.sum())
    sw_draw_neg = len(y_draw_train) / (2 * (len(y_draw_train) - y_draw_train.sum()))
    sw_draw = np.where(y_draw_train == 1, sw_draw_pos, sw_draw_neg)
    draw_model.fit(X_th_train, y_draw_train, sample_weight=sw_draw)

    # Stage 2: Home(0) vs Away(2) — remap to 0/1 for binary classifier
    non_draw_mask = y_th_train != 1
    y_ha_train = y_th_train[non_draw_mask].copy()
    y_ha_train[y_ha_train == 2] = 1  # remap: 0=home win, 1=away win

    ha_model = CalibratedClassifierCV(
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0),
        method='isotonic', cv=5
    )
    ha_sw = np.ones(non_draw_mask.sum())
    ha_counts = np.bincount(y_ha_train.astype(int), minlength=2)
    n_ha = non_draw_mask.sum()
    ha_sw[y_ha_train == 0] = n_ha / (2 * max(ha_counts[0], 1))
    ha_sw[y_ha_train == 1] = n_ha / (2 * max(ha_counts[1], 1))
    ha_model.fit(X_th_train[non_draw_mask], y_ha_train, sample_weight=ha_sw)

    # Find optimal draw threshold on validation set
    draw_proba_val = draw_model.predict_proba(X_th_val)[:, 1]
    ha_proba_val = ha_model.predict_proba(X_th_val)  # col0=P(home), col1=P(away)

    best_draw_thresh = 0.5
    best_acc = 0
    for dt in np.arange(0.1, 0.9, 0.02):
        preds = np.zeros(len(y_th_val), dtype=int)
        for j in range(len(y_th_val)):
            if draw_proba_val[j] > dt:
                preds[j] = 1  # draw
            else:
                # ha_proba: 0=home win, 1=away win → map back to 0, 2
                preds[j] = 0 if ha_proba_val[j, 0] > ha_proba_val[j, 1] else 2
        acc = accuracy_score(y_th_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_draw_thresh = dt

    print(f"  Optimal draw threshold: {best_draw_thresh:.2f}")
    print(f"  Two-stage validation accuracy: {best_acc:.4f}")

    # Apply to test set
    draw_proba_test = draw_model.predict_proba(X_test)[:, 1]
    ha_proba_test = ha_model.predict_proba(X_test)
    y_pred_2stage = np.zeros(len(y_test), dtype=int)
    for j in range(len(y_test)):
        if draw_proba_test[j] > best_draw_thresh:
            y_pred_2stage[j] = 1
        else:
            y_pred_2stage[j] = 0 if ha_proba_test[j, 0] > ha_proba_test[j, 1] else 2

    print(f"\nTwo-stage test accuracy: {accuracy_score(y_test, y_pred_2stage):.4f}")
    print("\n--- Two-Stage Classification Report ---")
    print(classification_report(y_test, y_pred_2stage, target_names=TARGET_NAMES, digits=3))

    # ------------------------------------------------------------------
    # Calibrated model with optimized per-class thresholds
    # ------------------------------------------------------------------
    print("\nFinding optimal thresholds on calibrated model...")
    # Use lower thresholds, especially for draws
    best_f1_cal = -1
    best_t_cal = np.array([0.35, 0.20, 0.35])
    cal_val_proba = cal_model.predict_proba(X_th_val)
    for t0 in np.arange(0.2, 0.6, 0.05):
        for t1 in np.arange(0.05, 0.45, 0.05):
            for t2 in np.arange(0.2, 0.6, 0.05):
                thresholds = np.array([t0, t1, t2])
                preds = np.zeros(len(y_th_val), dtype=int)
                for j in range(len(y_th_val)):
                    above = [c for c in range(3) if cal_val_proba[j, c] > thresholds[c]]
                    preds[j] = max(above, key=lambda c: cal_val_proba[j, c]) if above else np.argmax(cal_val_proba[j])
                f1 = f1_score(y_th_val, preds, average="macro", zero_division=0)
                acc = accuracy_score(y_th_val, preds)
                # Optimize for accuracy with minimum draw F1
                draw_f1 = f1_score(y_th_val, preds, average=None, zero_division=0)[1]
                if draw_f1 > 0.05 and acc > best_f1_cal:
                    best_f1_cal = acc
                    best_t_cal = thresholds.copy()
                elif best_f1_cal == -1 and f1 > best_f1_cal:
                    best_f1_cal = f1
                    best_t_cal = thresholds.copy()

    print(f"  Calibrated optimal thresholds: HW={best_t_cal[0]:.2f} D={best_t_cal[1]:.2f} AW={best_t_cal[2]:.2f}")

    # Apply to test set
    y_pred_cal_tuned = np.zeros(len(y_test), dtype=int)
    for j in range(len(y_test)):
        above = [c for c in range(3) if cal_proba_test[j, c] > best_t_cal[c]]
        y_pred_cal_tuned[j] = max(above, key=lambda c: cal_proba_test[j, c]) if above else np.argmax(cal_proba_test[j])

    print(f"  Calibrated + thresholded accuracy: {accuracy_score(y_test, y_pred_cal_tuned):.4f}")
    print(classification_report(y_test, y_pred_cal_tuned, target_names=TARGET_NAMES, digits=3))

    optimal_thresholds = best_t_cal

    # ------------------------------------------------------------------
    # Poisson + Dixon-Coles
    # ------------------------------------------------------------------
    poisson_home, poisson_away, poisson_fc, poisson_scaler, rho = train_poisson_models(results, features)

    # ------------------------------------------------------------------
    # Ensemble: calibrated classifier + Poisson
    # ------------------------------------------------------------------
    # Get probs on validation set for ensemble training
    cal_val_proba = cal_model.predict_proba(X_th_val)
    X_val_pois_raw = features.loc[thresh_val_mask, poisson_fc].values.astype(np.float64)
    X_val_pois = np.clip(poisson_scaler.transform(X_val_pois_raw), -5, 5)
    val_home_pred = poisson_home.predict(X_val_pois)
    val_away_pred = poisson_away.predict(X_val_pois)
    pois_val_proba = poisson_win_draw_loss_dc(val_home_pred, val_away_pred, rho)

    meta_model = train_ensemble(cal_val_proba, pois_val_proba, y_th_val)

    # Test set ensemble
    X_test_pois_raw = features.loc[test_mask, poisson_fc].values.astype(np.float64)
    X_test_pois = np.clip(poisson_scaler.transform(X_test_pois_raw), -5, 5)
    test_home_pred = poisson_home.predict(X_test_pois)
    test_away_pred = poisson_away.predict(X_test_pois)
    pois_test_proba = poisson_win_draw_loss_dc(test_home_pred, test_away_pred, rho)
    ensemble_test_proba = meta_model.predict_proba(np.hstack([cal_proba_test, pois_test_proba]))
    ensemble_preds = np.argmax(ensemble_test_proba, axis=1)

    # ------------------------------------------------------------------
    # Calibration plot (4 models)
    # ------------------------------------------------------------------
    print("\nGenerating calibration plots...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    model_data = [
        (best_cls_proba_test, f"{best_cls_name} (raw)", 0),
        (cal_proba_test, f"{best_cls_name} (calibrated)", 1),
        (pois_test_proba, "Poisson+DC", 2),
        (ensemble_test_proba, "Ensemble", 3),
    ]
    for proba, name, row in model_data:
        for i, tname in enumerate(TARGET_NAMES):
            y_binary = (y_test == i).astype(int)
            frac, mean = calibration_curve(y_binary, proba[:, i], n_bins=10)
            axes[row, i].plot(mean, frac, "s-", label=name)
            axes[row, i].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[row, i].set_xlabel("Mean predicted probability")
            axes[row, i].set_ylabel("Fraction of positives")
            axes[row, i].set_title(f"{name}: {tname}")
            axes[row, i].legend()
    plt.tight_layout()
    plt.savefig("calibration_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved calibration_plot.png")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importances = best_cls_model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    print("\nTop 20 Feature Importances:")
    for _, row in feat_imp.head(20).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top = feat_imp.head(20)
    ax.barh(range(20), top["importance"].values)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Save test predictions
    # ------------------------------------------------------------------
    test_preds = test[["date", "home_team", "away_team", "home_score", "away_score",
                        "tournament", "neutral"]].copy()
    test_preds["target"] = y_test
    test_preds["cal_prob_hw"] = cal_proba_test[:, 0]
    test_preds["cal_prob_d"] = cal_proba_test[:, 1]
    test_preds["cal_prob_aw"] = cal_proba_test[:, 2]
    test_preds["cal_pred_2stage"] = y_pred_2stage
    test_preds["pois_dc_prob_hw"] = pois_test_proba[:, 0]
    test_preds["pois_dc_prob_d"] = pois_test_proba[:, 1]
    test_preds["pois_dc_prob_aw"] = pois_test_proba[:, 2]
    test_preds["ensemble_prob_hw"] = ensemble_test_proba[:, 0]
    test_preds["ensemble_prob_d"] = ensemble_test_proba[:, 1]
    test_preds["ensemble_prob_aw"] = ensemble_test_proba[:, 2]
    test_preds.to_csv("predictions_test.csv", index=False)
    print(f"\n  Saved predictions_test.csv ({len(test_preds)} rows)")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    artifacts = {
        "model": cal_model,
        "model_name": best_cls_name,
        "feature_cols": feature_cols,
        "poisson_home": poisson_home,
        "poisson_away": poisson_away,
        "poisson_feature_cols": poisson_fc,
        "poisson_scaler": poisson_scaler,
        "dc_rho": rho,
        "meta_model": meta_model,
        "draw_model": draw_model,
        "ha_model": ha_model,
        "draw_threshold": best_draw_thresh,
        "optimal_thresholds": optimal_thresholds,
        "elo_history": dict(elo_history),
        "final_elos": final_elos,
        "name_map": name_map,
        "ELO_INITIAL": ELO_INITIAL,
        "ELO_K": ELO_K,
        "ELO_HOME_BOOST": ELO_HOME_BOOST,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("  Saved model.pkl")

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Test 2018-2024)")
    print("=" * 60)
    comparisons = [
        (f"{best_cls_name} (raw)", best_cls_proba_test),
        (f"{best_cls_name} (calibrated)", cal_proba_test),
        (f"{best_cls_name} (cal+threshold)", cal_proba_test),
        ("Two-Stage (draw+HA)", None),
        ("Poisson+DC", pois_test_proba),
        ("Ensemble", ensemble_test_proba),
    ]
    for name, proba in comparisons:
        if name == f"{best_cls_name} (cal+threshold)":
            preds = y_pred_cal_tuned
        elif proba is None:
            preds = y_pred_2stage
        else:
            preds = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, preds)
        if proba is not None:
            ll = log_loss(y_test, proba)
            f1 = f1_score(y_test, preds, average='macro')
            print(f"  {name:30s} Acc={acc:.4f}  LogLoss={ll:.4f}  MacroF1={f1:.4f}")
        else:
            f1 = f1_score(y_test, preds, average='macro')
            print(f"  {name:30s} Acc={acc:.4f}  MacroF1={f1:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
