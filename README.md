# International Football Match Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Docker](https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/ITZ41/international-football-predictor/ci.yml?style=flat-square&label=CI&logo=github)

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

Predictions are a weighted average of three independently-trained models. XGBoost (calibrated with isotonic regression) contributes the learned feature interactions and non-linear patterns. Poisson+Dixon-Coles models goal-scoring intensity with a dependence correction for low-scoring games. LogisticRegression provides a stable linear baseline. The weights are tuned on the validation set to minimize log loss, producing a final probability distribution over Home Win / Draw / Away Win.

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
python train.py              # trains & saves model.pkl
python predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral True
python simulate_tournament.py --groups sample_groups.csv --n_sims 10000 --output results.csv
streamlit run app.py         # dashboard at http://localhost:8501
```

## Docker

```bash
docker build -t football-predictor .
docker run -p 8501:8501 football-predictor
```

This builds the image and starts the Streamlit dashboard. No manual dependency install needed.

## Tests

```bash
python test_pipeline.py
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
app.py                      # Streamlit dashboard (4 tabs)
train.py                    # Model training, calibration, evaluation, artifact export
predict.py                  # Single-match + batch prediction CLI
simulate_tournament.py      # Monte Carlo tournament simulation
features.py                 # Feature engineering + Elo rating computation
test_pipeline.py            # End-to-end test suite

model.pkl                   # Trained model artifacts (~15MB)
results.csv                 # 49,398 matches (1872–2024)
elo_ratings.csv             # Time-series Elo ratings per team
goalscorers.csv             # Match-level goalscorer data
shootouts.csv               # Penalty shootout results
former_names.csv            # Historical → current team name mapping
world_cup_groups.csv        # World Cup group definitions for simulation
sample_groups.csv           # Example input for tournament simulator

MODEL_README.md             # Detailed model internals & usage guide
model_card.md               # Responsible AI model card
```

---

## Documentation

- [`MODEL_README.md`](MODEL_README.md) — Full model training walkthrough, hyperparameter details, file format descriptions, and advanced usage.
- [`model_card.md`](model_card.md) — Responsible AI card: intended use, performance per class, known limitations, and ethical considerations.

---

## Contributing

PRs welcome. Before submitting:

1. Run `python test_pipeline.py` — all checks must pass.
2. For new features: add corresponding tests to `test_pipeline.py`.
3. Keep the feature pipeline in `features.py` leakage-free (no future data).
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
