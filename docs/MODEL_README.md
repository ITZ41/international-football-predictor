# International Football Match Predictor

Predicts match outcomes (Home Win / Draw / Away Win) using Elo ratings,
form features, head-to-head records, and XGBoost.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python train.py
```

This produces:
- `model.pkl` — trained XGBoost model + artifacts
- `elo_ratings.csv` — current Elo ratings for all teams
- `predictions_test.csv` — test set (2018–2024) predictions
- `calibration_plot.png` — probability calibration charts
- `feature_importance.png` — feature importance chart

### 2. Predict a match

```bash
python predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral False
python predict.py --home "France" --away "Germany" --tournament "Friendly" --neutral True
```

Output includes:
- Win/Draw/Loss probabilities
- Elo ratings for both teams
- Head-to-head summary (last 10 meetings)
- Most likely scoreline (Poisson model)
- Predicted winner

## Files

| File | Purpose |
|------|---------|
| `features.py` | Feature engineering: Elo, form, H2H, rest, GD trend, tournament encoding |
| `train.py` | Model training, evaluation, calibration, artifact export |
| `predict.py` | CLI for single-match prediction |
| `requirements.txt` | Python dependencies |

## Model Details

- **Algorithm**: XGBoost (multi-class softmax)
- **Features**: 25+ including Elo diff, rolling form (5 & 10 match), H2H, rest days, GD trend, tournament type
- **Train/Test Split**: pre-2018 / 2018–2024
- **Hyperparameter tuning**: RandomizedSearchCV with time-series cross-validation
- **Team name handling**: former_names.csv maps historical names to current ones
- **No data leakage**: all features computed from pre-match data only
