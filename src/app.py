"""
Streamlit dashboard for international football match prediction.
Run:  streamlit run app.py

Tabs:
  1. Match Predictor — single match prediction with probabilities, heatmap, H2H
  2. Tournament Simulator — upload groups CSV, run Monte Carlo, see bracket
  3. Model Performance — calibration plot, feature importance
  4. Match History — H2H results between any two teams
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import io

from features import (
    load_data, build_name_resolver, standardize_results, encode_tournament,
    get_feature_columns, H2H_WINDOW, ELO_INITIAL,
)
from predict import (
    compute_all_features_for_match, dc_tau, poisson_dc_scorelines,
    MAX_GOALS,
)

# -- helpers --

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


@st.cache_data
def load_all():
    with open("models/model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    results, _, _, former = load_data("data")
    results = standardize_results(results, artifacts["name_map"])
    results = results.sort_values("date").reset_index(drop=True)
    elo_df = pd.read_csv("data/elo_ratings.csv")
    return artifacts, results, elo_df


def run_prediction(home_team, away_team, tournament, neutral, match_date, results, artifacts):
    """Run all three models and return results dict."""
    name_map = artifacts["name_map"]
    elo_history = artifacts["elo_history"]
    feature_cols = artifacts["feature_cols"]
    poisson_fc = artifacts["poisson_feature_cols"]
    poisson_scaler = artifacts["poisson_scaler"]
    rho = artifacts["dc_rho"]

    xgb_fd, pois_fd, home_elo, away_elo, h2h, tourney_type = \
        compute_all_features_for_match(
            home_team, away_team, tournament, neutral, match_date,
            results, elo_history, name_map
        )

    for col in feature_cols:
        if col.startswith("tourney_"):
            xgb_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
    X_xgb = np.array([[xgb_fd.get(col, 0) for col in feature_cols]])
    xgb_proba = artifacts["model"].predict_proba(X_xgb)[0]

    for col in poisson_fc:
        if col.startswith("tourney_"):
            pois_fd[col] = 1 if tourney_type == col.replace("tourney_", "") else 0
    X_pois = np.clip(poisson_scaler.transform(
        np.array([[pois_fd.get(col, 0) for col in poisson_fc]], dtype=np.float64)), -5, 5)
    lh = max(float(artifacts["poisson_home"].predict(X_pois)[0]), 0.01)
    la = max(float(artifacts["poisson_away"].predict(X_pois)[0]), 0.01)
    best_score, score_probs, p_hw, p_d, p_aw = poisson_dc_scorelines(lh, la, rho)

    w = artifacts["ensemble_weights"]
    ens_proba = w[0] * xgb_proba + w[1] * np.array([p_hw, p_d, p_aw])

    return {
        "xgb_proba": xgb_proba, "p_hw": p_hw, "p_d": p_d, "p_aw": p_aw,
        "ens_proba": ens_proba, "lh": lh, "la": la, "best_score": best_score,
        "score_probs": score_probs, "home_elo": home_elo, "away_elo": away_elo,
        "h2h": h2h, "tourney_type": tourney_type,
    }


# -- Streamlit UI --

st.set_page_config(page_title="Football Predictor", layout="wide")
st.title("International Football Match Predictor")

artifacts, results, elo_df = load_all()
teams = sorted(elo_df["team"].tolist())
tournaments = [
    "FIFA World Cup", "FIFA World Cup qualification", "Friendly",
    "UEFA Euro", "UEFA Euro qualification", "Copa América",
    "AFC Asian Cup", "Africa Cup of Nations", "CONCACAF Championship",
    "UEFA Nations League",
]

tab1, tab2, tab3, tab4 = st.tabs([
    "Match Predictor", "Tournament Simulator", "Model Performance", "Match History"
])

# ================================================================== #
# TAB 1: Match Predictor
# ================================================================== #
with tab1:
    st.header("Match Predictor")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

    col3, col4 = st.columns(2)
    with col3:
        tournament = st.selectbox("Tournament", tournaments)
    with col4:
        neutral = st.checkbox("Neutral Venue", value=True)

    if st.button("Predict", type="primary"):
        if home_team == away_team:
            st.error("Home and away team cannot be the same.")
        else:
            match_date = pd.Timestamp.now()
            pred = run_prediction(home_team, away_team, tournament, neutral, match_date, results, artifacts)

            st.subheader(f"{home_team} vs {away_team}")
            st.caption(f"{tournament} | {'Neutral' if neutral else 'Home venue'}")

            # Probability bar charts
            prob_data = pd.DataFrame({
                "Outcome": [f"{home_team} Win", "Draw", f"{away_team} Win"],
                "XGBoost": pred["xgb_proba"],
                "Poisson+DC": [pred["p_hw"], pred["p_d"], pred["p_aw"]],
                "Ensemble": pred["ens_proba"],
            })
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(3)
            width = 0.25
            ax.bar(x - width, prob_data["XGBoost"], width, label="XGBoost", color="#2196F3")
            ax.bar(x, prob_data["Poisson+DC"], width, label="Poisson+DC", color="#4CAF50")
            ax.bar(x + width, prob_data["Ensemble"], width, label="Ensemble", color="#FF9800")
            ax.set_xticks(x)
            ax.set_xticklabels(prob_data["Outcome"])
            ax.set_ylabel("Probability")
            ax.set_title("Win / Draw / Loss Probabilities")
            ax.legend()
            ax.set_ylim(0, 1)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)
            st.pyplot(fig)

            # Scoreline heatmap
            st.subheader("Scoreline Heatmap")
            heatmap_data = np.zeros((7, 7))
            for (h, a), p in pred["score_probs"].items():
                if h < 7 and a < 7:
                    heatmap_data[h, a] = p * 100
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
                        xticklabels=range(7), yticklabels=range(7),
                        ax=ax2, cbar_kws={"label": "Probability (%)"},
                        linewidths=0.5)
            ax2.set_xlabel(f"{away_team} Goals")
            ax2.set_ylabel(f"{home_team} Goals")
            ax2.set_title("P(home_goals, away_goals)")
            ax2.invert_yaxis()
            st.pyplot(fig2)

            # Summary metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Expected Goals (Home)", f"{pred['lh']:.2f}")
            col_b.metric("Expected Goals (Away)", f"{pred['la']:.2f}")
            col_c.metric(f"{home_team} Elo", f"{pred['home_elo']:.0f}")
            col_d.metric(f"{away_team} Elo", f"{pred['away_elo']:.0f}")

            # Top 5 scorelines
            st.subheader("Most Likely Scorelines")
            top5 = sorted(pred["score_probs"].items(), key=lambda x: -x[1])[:5]
            cols5 = st.columns(5)
            for idx, ((h, a), p) in enumerate(top5):
                cols5[idx].metric(f"{h}-{a}", f"{p*100:.1f}%")

            # H2H table
            name_map = artifacts["name_map"]
            home_r = name_map.get(home_team, home_team)
            away_r = name_map.get(away_team, away_team)
            pair_mask = (
                ((results["home_team"] == home_r) & (results["away_team"] == away_r)) |
                ((results["home_team"] == away_r) & (results["away_team"] == home_r))
            )
            h2h_matches = results[pair_mask]
            recent = h2h_matches.tail(10)

            if len(recent) > 0:
                st.subheader("Head-to-Head (last 10 meetings)")
                h2h_hw = h2h_d = h2h_aw = 0
                for _, m in recent.iterrows():
                    if m["home_score"] > m["away_score"]:
                        if m["home_team"] == home_r: h2h_hw += 1
                        else: h2h_aw += 1
                    elif m["home_score"] < m["away_score"]:
                        if m["away_team"] == away_r: h2h_aw += 1
                        else: h2h_hw += 1
                    else:
                        h2h_d += 1

                st.caption(f"{home_team}: {h2h_hw}W {h2h_d}D {h2h_aw}L")

                h2h_display = recent[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
                h2h_display["date"] = h2h_display["date"].dt.strftime("%Y-%m-%d")
                h2h_display.columns = ["Date", "Home", "Away", "H", "A", "Tournament"]
                h2h_display = h2h_display.reset_index(drop=True)

                # Color code rows from home team's perspective
                def color_row(row):
                    h, a = row["H"], row["A"]
                    home_is_home = row["Home"] == home_team
                    if home_is_home:
                        result = "W" if h > a else ("D" if h == a else "L")
                    else:
                        result = "W" if a > h else ("D" if h == a else "L")
                    colors = {"W": "#d4edda", "D": "#fff3cd", "L": "#f8d7da"}
                    return [f"background-color: {colors[result]}"] * len(row)

                st.dataframe(h2h_display.style.apply(color_row, axis=1), use_container_width=True)


# ================================================================== #
# TAB 2: Tournament Simulator
# ================================================================== #
with tab2:
    st.header("Tournament Simulator")
    st.markdown("Upload a CSV with `team` and `group` columns, then run Monte Carlo simulation.")

    uploaded = st.file_uploader("Upload groups.csv", type=["csv"])

    n_sims = st.slider("Number of simulations", 100, 50000, 10000, 100)

    if uploaded is not None and st.button("Simulate Tournament", type="primary"):
        groups_df = pd.read_csv(uploaded)
        if "team" not in groups_df.columns or "group" not in groups_df.columns:
            st.error("CSV must have 'team' and 'group' columns")
        else:
            groups = {}
            for _, row in groups_df.iterrows():
                g = row["group"].strip().upper()
                groups.setdefault(g, []).append(row["team"])

            st.write("Groups:")
            for g, t in sorted(groups.items()):
                st.write(f"  Group {g}: {', '.join(t)}")

            from simulate_tournament import simulate_tournament, precompute_match_probs

            all_teams_sorted = sorted({t for g in groups.values() for t in g})
            match_date = pd.Timestamp("2026-07-01")

            with st.spinner("Pre-computing match probabilities..."):
                prob_table = precompute_match_probs(all_teams_sorted, artifacts, results, match_date)

            with st.spinner(f"Running {n_sims} simulations..."):
                sim_results = simulate_tournament(groups, artifacts, results, n_sims)

            # Display results table
            stages = ["Group", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
            st.subheader("Simulation Results")
            st.dataframe(sim_results.style.format({s: "{:.1f}%" for s in stages}),
                         use_container_width=True)

            # Win probability bar chart
            chart_data = sim_results.sort_values("Winner", ascending=True).tail(16)
            GROUP_COLORS = {
                "A": "#e41a1c", "B": "#377eb8", "C": "#4daf4a", "D": "#984ea3",
                "E": "#ff7f00", "F": "#ffff33", "G": "#a65628", "H": "#f781bf",
            }
            team_group = {}
            for g, ts in groups.items():
                for t in ts:
                    team_group[t] = g
            colors = [GROUP_COLORS.get(team_group.get(t, ""), "#999999") for t in chart_data["team"]]

            fig, ax = plt.subplots(figsize=(10, max(6, len(chart_data) * 0.4)))
            ax.barh(chart_data["team"], chart_data["Winner"], color=colors)
            ax.set_xlabel("Win Probability (%)")
            ax.set_title("Tournament Win Probability by Team")
            st.pyplot(fig)

            # Top contenders
            st.subheader("Top Contenders")
            top5 = sim_results.nlargest(5, "Winner")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                st.write(f"  {i}. **{row['team']}** — Winner: {row['Winner']:.1f}% | "
                         f"Final: {row['Final']:.1f}%")

            # Download results
            csv = sim_results.to_csv(index=False)
            st.download_button("Download Results CSV", csv, "tournament_results.csv", "text/csv")


# ================================================================== #
# TAB 3: Model Performance
# ================================================================== #
with tab3:
    st.header("Model Performance")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.subheader("Calibration Plot")
        cal_path = "assets/calibration_plot.png"
        try:
            cal_img = plt.imread(cal_path)
            fig, ax = plt.subplots(figsize=(12, 16))
            ax.imshow(cal_img)
            ax.axis("off")
            st.pyplot(fig)
        except FileNotFoundError:
            st.warning("calibration_plot.png not found. Run train.py to generate.")

    with perf_col2:
        st.subheader("Feature Importance")
        fi_path = "assets/feature_importance.png"
        try:
            fi_img = plt.imread(fi_path)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(fi_img)
            ax.axis("off")
            st.pyplot(fig)
        except FileNotFoundError:
            st.warning("feature_importance.png not found. Run train.py to generate.")

    # Metrics table
    st.subheader("Performance Summary")
    metrics = pd.DataFrame({
        "Model": ["XGBoost (calibrated)", "Poisson+DC", "Ensemble"],
        "Accuracy": ["59.6%", "59.2%", "58.8%"],
        "Log Loss": ["0.879", "0.909", "0.890"],
        "Brier Score": ["0.173", "—", "—"],
    })
    st.table(metrics)


# ================================================================== #
# TAB 4: Match History
# ================================================================== #
with tab4:
    st.header("Match History (Head-to-Head)")

    col1, col2 = st.columns(2)
    with col1:
        h2h_home = st.selectbox("Team 1", teams, index=teams.index("Brazil") if "Brazil" in teams else 0, key="h2h_home")
    with col2:
        h2h_away = st.selectbox("Team 2", teams, index=teams.index("Argentina") if "Argentina" in teams else 1, key="h2h_away")

    if st.button("Show History", key="h2h_btn"):
        if h2h_home == h2h_away:
            st.error("Select two different teams.")
        else:
            name_map = artifacts["name_map"]
            t1 = name_map.get(h2h_home, h2h_home)
            t2 = name_map.get(h2h_away, h2h_away)

            pair_mask = (
                ((results["home_team"] == t1) & (results["away_team"] == t2)) |
                ((results["home_team"] == t2) & (results["away_team"] == t1))
            )
            h2h = results[pair_mask].sort_values("date", ascending=False).head(20)

            if len(h2h) == 0:
                st.info(f"No matches found between {h2h_home} and {h2h_away}.")
            else:
                # Summary stats
                hw = d = aw = 0
                total_goals = 0
                for _, m in h2h.iterrows():
                    total_goals += m["home_score"] + m["away_score"]
                    if m["home_score"] > m["away_score"]:
                        if m["home_team"] == t1: hw += 1
                        else: aw += 1
                    elif m["home_score"] < m["away_score"]:
                        if m["away_team"] == t2: aw += 1
                        else: hw += 1
                    else:
                        d += 1

                st.subheader(f"{h2h_home} vs {h2h_away} — All Meetings")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric(f"{h2h_home} Wins", hw)
                sc2.metric("Draws", d)
                sc3.metric(f"{h2h_away} Wins", aw)
                sc4.metric("Avg Goals", f"{total_goals / len(h2h):.1f}")

                # Results table
                display = h2h[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
                display["date"] = display["date"].dt.strftime("%Y-%m-%d")
                display.columns = ["Date", "Home", "Away", "H", "A", "Tournament"]
                display = display.reset_index(drop=True)

                # Color code from Team 1's perspective
                def color_h2h(row):
                    h, a = row["H"], row["A"]
                    t1_is_home = row["Home"] == h2h_home
                    if t1_is_home:
                        result = "W" if h > a else ("D" if h == a else "L")
                    else:
                        result = "W" if a > h else ("D" if h == a else "L")
                    colors = {"W": "#d4edda", "D": "#fff3cd", "L": "#f8d7da"}
                    return [f"background-color: {colors[result]}"] * len(row)

                st.dataframe(display.style.apply(color_h2h, axis=1), use_container_width=True)

                # Timeline chart
                if len(h2h) >= 3:
                    st.subheader("Goals Over Time")
                    timeline_data = h2h.sort_values("date")[["date", "home_score", "away_score"]].copy()
                    timeline_data["date"] = timeline_data["date"].dt.strftime("%Y-%m-%d")
                    timeline_data = timeline_data.rename(columns={
                        "home_score": "Home Goals", "away_score": "Away Goals"
                    })
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(timeline_data["date"], timeline_data["Home Goals"], "bo-", label="Home Goals")
                    ax.plot(timeline_data["date"], timeline_data["Away Goals"], "ro-", label="Away Goals")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Goals")
                    ax.legend()
                    ax.tick_params(axis="x", rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)


st.markdown("---")
st.caption("Models: XGBoost + Poisson-DixonColes + LogisticRegression Ensemble | "
           "Features: 44 | Training: pre-2018 | Test: 2018-2024")
