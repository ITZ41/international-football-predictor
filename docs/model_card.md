# Model Card: International Football Match Predictor

## Model Details

- **Name**: International Football Match Outcome Predictor
- **Version**: 3.0
- **Type**: Multi-class classification (Home Win / Draw / Away Win)
- **Primary Model**: XGBoost with isotonic calibration
- **Secondary Models**: LightGBM, Poisson+Dixon-Coles, LogisticRegression Ensemble
- **Training Date**: 2024
- **Framework**: scikit-learn, XGBoost, LightGBM

## Purpose

Predicts the outcome of international football (soccer) matches between national teams.
Outputs probabilities for three outcomes: Home Win, Draw, Away Win.

**Intended use**: Tournament simulation, match preview analysis, understanding team strength dynamics.

**NOT intended for**: Betting advice, gambling, or any financial decision-making.

## Training Data

| Property | Value |
|----------|-------|
| Source | results.csv (international football results) |
| Date Range | ~1872 – 2024 |
| Total Matches | ~49,500 |
| Teams | 336 |
| Training Set | Matches before 2018 (~38,000) |
| Validation Set | 2014–2017 (~4,700) |
| Test Set | 2018–2024 (~6,800) |

## Performance Metrics (Test Set: 2018–2024)

| Model | Accuracy | Log Loss | Macro F1 | Brier Score |
|-------|----------|----------|----------|-------------|
| XGBoost (raw) | 49.1% | 0.960 | 0.494 | — |
| XGBoost (calibrated) | **59.6%** | **0.879** | 0.439 | 0.173 |
| XGBoost (cal + threshold) | 59.0% | 0.879 | 0.480 | — |
| LightGBM (raw) | 49.0% | — | — | — |
| Poisson+Dixon-Coles | 59.2% | 0.909 | 0.431 | — |
| Ensemble | 58.8% | 0.890 | 0.475 | — |

### Per-Class Performance (Calibrated XGBoost + Thresholds)

| Outcome | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Home Win | 62.1% | 85.3% | 71.9% |
| Draw | 31.1% | 10.0% | 15.1% |
| Away Win | 59.2% | 55.0% | 57.0% |

### 2022 World Cup Top-2 Accuracy

On 5 key group stage matches: **80%** (4/5 correct in top-2 predictions).

## Feature Descriptions (44 features)

### Top Features by Importance

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `elo_diff` | Difference in Elo ratings (home - away) |
| 2 | `elo_pct_diff` | Elo percentile rank difference |
| 3 | `home_advantage` | Binary: 1 if home venue, 0 if neutral |
| 4 | `home_gc_avg_10` | Home team goals conceded (last 10) |
| 5 | `away_gc_avg_10` | Away team goals conceded (last 10) |
| 6 | `h2h_home_win_rate` | Head-to-head home win rate |
| 7 | `tourney_qualifier` | Tournament is a qualifier |
| 8 | `away_elo_pct` | Away team Elo percentile |
| 9 | `away_gd_trend_5` | Away team goal difference trend (last 5) |
| 10 | `home_gd_trend_5` | Home team goal difference trend (last 5) |

### Feature Categories

- **Elo-based** (5): home_elo, away_elo, elo_diff, home_elo_pct, away_elo_pct, elo_pct_diff
- **Form** (12): Win rate, goals scored/conceded at 5 and 10 match windows
- **Home/Away Split** (4): Separate home and away win rates
- **Head-to-Head** (3): H2H win rate, draw rate, average goals
- **Context** (4): Rest days, goal difference trend, home advantage
- **Advanced** (8): Competitive ratio, streak, clean sheet rate, last-3 scoring
- **Tournament** (6+): One-hot encoded tournament type

## Known Limitations

1. **Draw Prediction**: The model struggles with draw prediction (F1 ~15%). Draws are inherently harder to predict in football due to their dependence on tactical decisions, game state, and motivation.

2. **Friendly vs. Competitive Gap**: Friendly matches have different dynamics (experimental lineups, lower intensity). The model partially accounts for this via tournament encoding and competitive ratio features.

3. **Data Cutoff**: Training data has a cutoff. Recent team changes (new coaches, player retirements, emerging talent) are not captured.

4. **No Player-Level Data**: The model uses only team-level aggregate features. Injuries, suspensions, and individual player form are not modeled.

5. **No Tactical Data**: Formation, playing style, and tactical matchups are not captured.

6. **Home/Away Assignment**: In neutral-venue tournaments (World Cups), the "home" team is often arbitrary and may not reflect true home advantage.

7. **Class Imbalance**: Home wins (~46%) are more common than draws (~23%) or away wins (~31%). The model uses class weights to partially address this.

## How to Run

### Train
```bash
python train.py
```

### Predict Single Match
```bash
python predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral True
```

### Batch Predict
```bash
python predict.py --batch fixtures.csv --output predictions.csv
```

### Simulate Tournament
```bash
python simulate_tournament.py --groups groups.csv --n_sims 10000 --output results.csv
```

### Run Dashboard
```bash
streamlit run app.py
```

### Run Tests
```bash
python test_pipeline.py
```

## Ethical Considerations

- **NOT for gambling**: This model should not be used for betting or gambling purposes. Football outcomes are inherently probabilistic, and the model's accuracy (~60%) means it is wrong ~40% of the time.

- **No guarantee of fairness**: Historical data reflects historical biases in refereeing, scheduling, and competition structure.

- **Unintended consequences**: Publicly available prediction models can affect betting markets and potentially enable manipulation if misused.

- **Scope**: This model is designed for analysis and entertainment. It should not be the sole basis for any consequential decision.

## Changelog

### v3.0 (Current)
- Added LightGBM model (compares with XGBoost, keeps best)
- Added isotonic calibration (CalibratedClassifierCV)
- Added two-stage draw prediction
- Added last-3 scoring trend feature
- Added Elo percentile features
- Fixed prediction-time feature leak (all features computed from historical data)
- Fixed clean sheet rate bug (was using home_cs for both columns)
- Tournament simulator: pre-computed probability table (10K sims in ~5s)
- Tournament simulator: visual output (bar chart + formatted table)
- Streamlit dashboard: tournament tab, performance tab, match history tab

### v2.0
- Dixon-Coles correction for Poisson model
- LogisticRegression ensemble
- Threshold tuning for draw prediction
- Calibration plots
- Batch prediction mode

### v1.0
- Initial XGBoost + Poisson pipeline
- Elo rating system
- Basic feature engineering
