# International Football Match Predictor

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Docker](https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/ITZ41/international-football-predictor/ci.yml?style=flat-square&label=CI&logo=github)

Predict outcomes of international football matches using ensemble of XGBoost, Poisson+Dixon-Coles, and LogisticRegression models. Includes Monte Carlo tournament simulator and Streamlit dashboard.

## Models

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| XGBoost (calibrated) | 59.6% | 0.879 |
| Poisson+Dixon-Coles | 59.2% | 0.909 |
| Ensemble | 58.8% | 0.890 |

44 features: Elo ratings, form (5/10 match windows), H2H, streak, clean sheet rate, competitive ratio, last-3 scoring, Elo percentile, tournament one-hot.

Train: pre-2018 (~38K) | Validation: 2014-2017 (~4.7K) | Test: 2018-2024 (~6.8K)

## Quick Start

```bash
pip install -r requirements.txt
python train.py              # trains model.pkl
python predict.py --home "Brazil" --away "Argentina" --tournament "FIFA World Cup" --neutral True
python simulate_tournament.py --groups sample_groups.csv --n_sims 10000 --output results.csv
streamlit run app.py         # dashboard at http://localhost:8501
```

## Docker

```bash
docker build -t football-predictor .
docker run -p 8501:8501 football-predictor
```

## Tests

```bash
python test_pipeline.py
```

10 checks: model artifacts, predictions on 2022 WC matches (80% top-2 accuracy), tournament simulator. Exits 0 on pass.

## Project Structure

```
app.py                    # Streamlit dashboard (4 tabs)
train.py                  # XGBoost + LightGBM + calibration + Poisson + ensemble
predict.py                # Single match + batch prediction
simulate_tournament.py    # Monte Carlo tournament simulation
features.py               # Feature engineering, Elo ratings
test_pipeline.py          # End-to-end test suite
model.pkl                 # Trained model artifacts (~15MB)
results.csv               # 49,398 international match results (1872-2024)
```

## License

MIT — see [LICENSE](LICENSE).
