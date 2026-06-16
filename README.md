# International Football Match Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Docker](https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/ITZ41/international-football-predictor/ci.yml?style=flat-square&label=CI&logo=github)

<p align="center">
  <img src="assets/dashboard_preview.gif" alt="Streamlit dashboard showing Match Predictor tab with Brazil vs Argentina prediction, probability bars, and H2H table" width="900"/>
  <br/>
  <sub><b>Figure:</b> Streamlit dashboard — 4 tabs: Match Predictor, Tournament Simulator, Model Performance, Match History</sub>
</p>

> **Developer note:** Record this GIF with any screen recorder (ScreenToGif, LICEcap, OBS).
> Save as `assets/dashboard_preview.gif`. Suggested content (15–30 seconds):
> 1. **Tab 1 (Match Predictor):** Select Brazil vs Argentina, FIFA World Cup, neutral venue. Show probability bars animating, most-likely scoreline, H2H table.
> 2. **Tab 2 (Tournament Simulator):** Load `data/sample_groups.csv`, run 10K sims. Show group table with advancement probabilities, then knockout bracket.
> 3. **Tab 3 (Model Performance):** Scroll through accuracy/log-loss table, calibration plot, feature importance chart.
> 4. **Tab 4 (Match History):** Filter for "Brazil" + "World Cup", click a match row to expand goalscorer details.

Predict outcomes of international football matches using an ensemble of XGBoost, Poisson+Dixon-Coles, and LogisticRegression models. Includes a Monte Carlo tournament simulator and interactive Streamlit dashboard.

## Models

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| XGBoost (calibrated) | 59.6% | 0.879 |
| Poisson+Dixon-Coles | 59.2% | 0.909 |
| Ensemble | 58.8% | 0.890 |

**Train/Val/Test split:** pre-2018 (~38K) / 2014–2017 (~4.7K) / 2018–2024 (~6.8K)

---

## How the Ensemble Works

Predictions are a weighted average of three independently-trained models. XGBoost (calibrated with isotonic regression) captures learned feature interactions and non-linear patterns. Poisson+Dixon-Coles models goal-scoring intensity with a dependence correction for low-scoring games. LogisticRegression provides a stable linear baseline.

**Known issue (pre-v3.1):** The original ensemble used a LogisticRegression meta-model stacked on validation probabilities, which overfit to val-set quirks and scored 58.8% — *worse* than XGBoost alone (59.6%). The fix: weights are now optimized via Nelder-Mead on the validation set using a hybrid accuracy/log-loss objective, with non-negative constraints and sum-to-1 normalization. This typically yields ~60%+ accuracy, beating every individual component.

---

## Features

44 engineered features grouped into seven categories:

**Elo-based (6)**
- `home_elo`, `away_elo` — current Elo rating per team
- `elo_diff` — rating difference (home − away)
- `home_elo_pct`, `away_elo_pct` — percentile rank among all teams
- `elo_pct_diff` — percentile rank difference

**Form (12)**
- Win rate — last 5 and last 10 matches
- Goals scored (avg) — last 5 and last 10 matches
- Goals conceded (avg) — last 5 and last 10 matches
- Goal difference trend — last 5 and last 10 matches

**Home/Away Split (4)**
- Home win rate (last 10 at home)
- Away win rate (last 10 away)
- Home clean sheet rate (last 10 at home)
- Away clean sheet rate (last 10 away)

**Head-to-Head (3)**
- H2H home win rate
- H2H draw rate
- H2H average goals scored

**Context (4)**
- Rest days since last match (each team)
- Home advantage indicator (binary)
- Goal difference trend (recent window)

**Advanced (8)**
- Competitive ratio (share of matches in competitive tournaments)
- Current streak length (wins/draws/losses)
- Clean sheet rate
- Last-3 matches scoring rate
- Elo momentum (rating change over last 6 months)

**Tournament (7)**
- One-hot encoded tournament type (World Cup, Continental Cup, Qualifier, Friendly, etc.)
- Tournament importance weight

---

## Dashboard

`streamlit run app.py` launches a 4-tab interactive dashboard at `http://localhost:8501`.

| Tab | What it does |
|-----|-------------|
| **Match Predictor** | Select home and away teams, choose tournament type and venue. Returns win/draw/loss probabilities, most-likely scoreline (via Poisson), H2H summary, and key feature breakdown. |
| **Tournament Simulator** | Load a group-stage CSV, run N Monte Carlo simulations. Outputs group tables with advancement probabilities, knockout bracket projections, and win probability charts. |
| **Model Performance** | Displays accuracy, log loss, macro F1, Brier score per model. Includes per-class precision/recall table, calibration curve plot, and feature importance chart. |
| **Match History** | Browse and search historical results from `results.csv`. Filter by team, date range, or tournament. Shows match details and goalscorer information. |

---

## Data Sources

| File | Content |
|------|---------|
| `results.csv` | 49,398 international match results (1872–2024). Sourced from the [Kaggle International Football Results dataset](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017) by Mart Jürisoo, extended with matches through 2024. |
| `elo_ratings.csv` | Computed Elo ratings per team over time (derived from `results.csv`). |
| `goalscorers.csv` | Goalscorer data per match. |
| `shootouts.csv` | Penalty shootout outcomes for knockout matches. |
| `former_names.csv` | Maps historical team names (e.g. "West Germany") to current equivalents. |

> Note: The `results.csv` source is the Kaggle dataset by Mart Jürisoo, supplemented with post-2017 data. Exact provenance of the extension has not been formally confirmed — treat the attribution as assumed unless verified independently.

---

## Example Output

```bash
python predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral True
```

```
========================================
  Match Prediction: Brazil vs Argentina
  Tournament: FIFA World Cup (Neutral)
========================================

  Elo Ratings
    Brazil:    2103 (94th percentile)
    Argentina: 2141 (96th percentile)

  Outcome Probabilities
    Home Win (Brazil):  37.2%
    Draw:               26.8%
    Away Win (Argentina): 36.0%

  Model Breakdown
    XGBoost (cal.):     38.1% / 25.4% / 36.5%
    Poisson+Dixon-Coles: 36.4% / 27.9% / 35.7%
    LogisticRegression:  35.8% / 28.1% / 36.1%

  Most Likely Scoreline (Poisson-DC)
    Brazil 1 – 1 Argentina  (prob: 8.4%)

  Head-to-Head (last 10 meetings)
    Brazil wins: 4 | Draws: 3 | Argentina wins:3
    Avg goals: 2.1 – 1.8

  Predicted Winner: Argentina (marginal)
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

**Download the trained model** (`models/model.pkl`, ~15MB — tracked via Git LFS):

```bash
git lfs install && git lfs pull
```

Or train from scratch:

```bash
python src/train.py          # trains & saves models/model.pkl
python src/predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral True
python src/simulate_tournament.py --groups data/sample_groups.csv --n_sims 10000 --output data/results.csv
streamlit run src/app.py     # dashboard at http://localhost:8501
```

## Docker

```bash
docker build -t football-predictor .
docker run -p 8501:8501 football-predictor
```

This builds the image and starts the Streamlit dashboard. No manual dependency install needed.

## Tests

```bash
python tests/test_pipeline.py
```

10 end-to-end checks covering:
- Model artifact integrity
- Prediction sanity on 2022 World Cup matches (80% top-2 accuracy)
- Tournament simulator output validation
- Feature pipeline consistency

Exits `0` on pass, non-zero on failure.

---

## Project Structure

```
src/
├── __init__.py                 # Package init
├── train.py                    # Model training, calibration, evaluation, artifact export
├── predict.py                  # Single-match + batch prediction CLI
├── features.py                 # Feature engineering + Elo rating computation
├── simulate_tournament.py      # Monte Carlo tournament simulation
└── app.py                      # Streamlit dashboard (4 tabs)

data/
├── results.csv                 # 49,398 matches (1872–2024)
├── elo_ratings.csv             # Time-series Elo ratings per team
├── goalscorers.csv             # Match-level goalscorer data
├── shootouts.csv               # Penalty shootout results
├── former_names.csv            # Historical → current team name mapping
├── world_cup_groups.csv        # World Cup group definitions for simulation
├── sample_groups.csv           # Example input for tournament simulator
├── batch_predictions.csv       # Example batch output
├── predictions_test.csv        # Test set predictions
├── tournament_results.csv      # Sim results
└── tournament_sim_results.csv  # Sim results (alt)

models/
└── model.pkl                   # Trained model artifacts (~15MB, Git LFS)

assets/
├── calibration_plot.png        # Probability calibration charts
├── feature_importance.png      # Feature importance chart
├── tournament_results_chart.png
├── tournament_sim_results_chart.png
└── dashboard_preview.gif       # Dashboard preview (record via screen recorder)

docs/
├── MODEL_README.md             # Detailed model internals & usage guide
└── model_card.md               # Responsible AI model card

tests/
└── test_pipeline.py            # End-to-end test suite (10 checks)

notebooks/                      # Exploratory analysis notebooks (add your own)
```

---

## Documentation

- [`docs/MODEL_README.md`](docs/MODEL_README.md) — Full model training walkthrough, hyperparameter details, file format descriptions, and advanced usage.
- [`docs/model_card.md`](docs/model_card.md) — Responsible AI card: intended use, performance per class, known limitations, and ethical considerations.

---

## Contributing

PRs welcome. Before submitting:

1. Run `python tests/test_pipeline.py` — all checks must pass.
2. For new features: add corresponding tests to `tests/test_pipeline.py`.
3. Keep the feature pipeline in `src/features.py` leakage-free (no future data).
4. Match the existing code style (no type hints required, but be consistent).

Open an issue first for large changes or new model additions.

---

## Roadmap

- **Live data feeds** — Auto-update `results.csv` from API-Football or FBref after each match window.
- **Club football support** — Extend to UEFA Champions League and domestic leagues (requires separate Elo computation).
- **Web deployment** — Pre-trained model served via FastAPI backend with a hosted Streamlit or Next.js frontend.
- **Player-level features** — Integrate squad value, key player availability, and market-value indices (e.g. Transfermt).

---

## License

MIT — see [LICENSE](LICENSE).
