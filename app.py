# =====================================================================
#  Productivity Prediction Dashboard  (CatBoost + Streamlit)
#  Upload to GitHub (same folder): app.py, requirements.txt,
#  everyday_life_dataset.csv and the files from app_files.zip
#  (model.joblib, model.cbm, metrics.json, test_predictions.csv).
# =====================================================================
import io
import os
import json
import datetime
from math import erf, sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
from catboost import CatBoostRegressor
import streamlit as st

DATA_FILE = "everyday_life_dataset.csv"
TARGET = "productivity_rating"

NUM_BASE = ["age", "temperature_c", "sleep_hours", "screen_time_hours",
            "steps", "exercise_minutes", "caffeine_mg", "mood_score", "bedtime_hr"]
CAT_COLS = ["gender", "occupation", "chronotype", "climate_zone",
            "weather", "day_of_week", "user_id"]
# habits we summarise per user (their "normal")
USER_NUM = ["sleep_hours", "screen_time_hours", "caffeine_mg", "mood_score",
            "steps", "exercise_minutes", "bedtime_hr", "temperature_c"]
SKEWED = ["steps", "caffeine_mg", "exercise_minutes", "screen_time_hours"]


# ---------------------------------------------------------------------
# 1. DATA + FEATURES
# ---------------------------------------------------------------------
def bedtime_to_hours(value):
    """'02:10' -> 26.17  (hours after midnight of the previous day, so
    00:30 counts as LATER than 23:00 instead of earlier)."""
    if pd.isna(value):
        return np.nan
    h, m = str(value).split(":")[:2]
    x = int(h) + int(m) / 60
    return x + 24 if x < 18 else x


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["bedtime_hr"] = df["bedtime"].map(bedtime_to_hours)
    return df


def compute_user_stats(df):
    """Each user's usual habits (average and spread). Needs a 'user_id' column."""
    g = df.groupby("user_id")[USER_NUM]
    return g.mean().add_prefix("u_").join(g.std().add_prefix("ustd_"))


def make_features(df, user_stats):
    """Turns raw rows into model inputs. Used for training AND predicting.
    A user_id that is not in user_stats (a brand-new user) gets empty habit
    columns, which CatBoost handles natively."""
    X = pd.DataFrame(index=df.index)
    for c in NUM_BASE:
        X[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Variable transformation: log1p tames skewed columns ---
    for c in SKEWED:
        X["log_" + c] = np.log1p(X[c].clip(lower=0))

    # --- Engineered features ---
    X["sleep_dev"] = X["sleep_hours"] - 7.5            # distance from ideal sleep
    X["sleep_sq"] = X["sleep_dev"] ** 2                # too little AND too much hurts
    X["caffeine_per_sleep"] = X["caffeine_mg"] / (X["sleep_hours"] + 1)
    X["screen_per_sleep"] = X["screen_time_hours"] / (X["sleep_hours"] + 1)
    X["active_score"] = X["steps"].fillna(0) / 1000 + X["exercise_minutes"].fillna(0) / 10
    X["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # --- Personal baseline: how today compares with this user's normal ---
    stats = user_stats.reindex(df["user_id"].values)
    stats.index = df.index
    X = X.join(stats)
    for c in USER_NUM:
        X["dev_" + c] = X[c] - X["u_" + c]

    for c in CAT_COLS:
        X[c] = df[c].fillna("Unknown").astype(str)
    return X


# ---------------------------------------------------------------------
# 2. LOAD THE MODEL TRAINED IN GOOGLE COLAB
#    (files created by productivity_model_colab.ipynb)
# ---------------------------------------------------------------------
REQUIRED_FILES = [DATA_FILE, "metrics.json", "test_predictions.csv"]


def load_model_file():
    """Loads model.joblib; falls back to model.cbm if joblib can't read it."""
    if os.path.exists("model.joblib"):
        try:
            return joblib.load("model.joblib")
        except Exception:
            pass
    model = CatBoostRegressor()
    model.load_model("model.cbm")
    return model


def load_artifacts():
    with open("metrics.json") as f:
        m = json.load(f)
    test = pd.read_csv("test_predictions.csv")
    y_test = test["actual"]
    pred = test["predicted"].to_numpy()
    return {
        "model": load_model_file(),
        "X_test": test.drop(columns=["actual", "predicted"]),
        "y_test": y_test,
        "pred": pred,
        "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"],
        "importance": pd.Series(m["importance"]).sort_values(ascending=False),
        "n_train": m["n_train"], "n_test": m["n_test"],
        "best_iteration": m["best_iteration"],
        "params": m["params"],
        "cv_r2": m["cv_r2"],
        "train_r2": m["train_r2"],
        "explained_variance": m["explained_variance"],
        "feature_variance": m["feature_variance"],
        "learning_curve": m["learning_curve"],
        "residual_checks": m["residual_checks"],
        "delta_test": m["delta_test"],
        "trained_on": m.get("trained_on", "?"),
        "catboost_version": m.get("catboost_version", "?"),
    }


# ---------------------------------------------------------------------
# 3. GRAPHS  (each returns a matplotlib figure)
# ---------------------------------------------------------------------
BLUE, ORANGE, GREY = "#2f6fdb", "#e8743b", "#8a8f98"


def plot_actual_vs_pred(y, p, r2, size=(4.6, 3.7)):
    rng = np.random.default_rng(0)
    n = min(2500, len(y))
    idx = rng.choice(len(y), n, replace=False)
    yy = np.asarray(y)[idx] + rng.normal(0, 0.08, n)   # jitter: ratings are whole numbers
    pp = np.asarray(p)[idx]
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(yy, pp, s=7, alpha=0.3, color=BLUE)
    ax.plot([1, 10], [1, 10], "--", color=ORANGE, label="Perfect prediction")
    ax.set_xlabel("Actual productivity")
    ax.set_ylabel("Predicted productivity")
    ax.set_title(f"Actual vs Predicted  (R² = {r2:.3f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_residuals(y, p, size=(4.6, 3.7)):
    resid = np.asarray(y) - np.asarray(p)
    rng = np.random.default_rng(1)
    n = min(2500, len(resid))
    idx = rng.choice(len(resid), n, replace=False)
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(np.asarray(p)[idx], resid[idx], s=7, alpha=0.3, color=BLUE)
    ax.axhline(0, color=ORANGE, linestyle="--")
    ax.set_xlabel("Predicted productivity")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residual Plot  (mean {resid.mean():.2f}, std {resid.std():.2f})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_importance(importance, top=12, size=(4.6, 3.9)):
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=size)
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Feature Importance (top {len(imp)})", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_transformations(df, size=(4.6, 4.6)):
    """Before / after log transformation for two skewed variables."""
    cols = [("steps", "Steps"), ("caffeine_mg", "Caffeine (mg)")]
    fig, axes = plt.subplots(2, 2, figsize=size)
    for j, (c, label) in enumerate(cols):
        raw = df[c].dropna()
        logv = np.log1p(raw.clip(lower=0))
        axes[0, j].hist(raw, bins=40, color=GREY)
        axes[0, j].set_title(f"{label}\nskew = {raw.skew():.2f}", fontsize=8)
        axes[1, j].hist(logv, bins=40, color=BLUE)
        axes[1, j].set_title(f"log({label}) skew = {logv.skew():.2f}", fontsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    fig.suptitle("Variable Transformation (top: raw, bottom: log1p)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cv_scores(cv, size=(4.6, 3.4)):
    """R² on each of the cross-validation folds (stable model = similar bars)."""
    cv = list(cv)
    fig, ax = plt.subplots(figsize=size)
    ax.bar([f"Fold {i + 1}" for i in range(len(cv))], cv, color=BLUE)
    ax.axhline(np.mean(cv), color=ORANGE, linestyle="--",
               label=f"Mean {np.mean(cv):.3f} (std {np.std(cv):.3f})")
    ax.set_ylim(max(0, min(cv) - 0.05), min(1, max(cv) + 0.05))
    ax.set_ylabel("R²")
    ax.set_title("Cross-validation R² per fold", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_variance(fv, size=(5.5, 4.4)):
    """Variance of each feature after scaling to 0-1 (near 0 = carries no information)."""
    s = pd.Series(fv).sort_values()
    colors = [ORANGE if v < 0.01 else BLUE for v in s.values]
    fig, ax = plt.subplots(figsize=size)
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0.01, color="black", linestyle="--", linewidth=0.8, label="Low-variance limit (0.01)")
    ax.set_xlabel("Variance of feature (scaled 0-1)")
    ax.set_title("Feature variance check", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_learning_curve(n_rows, r2s, size=(4.6, 3.4)):
    """R² on the test set when the model is trained on more and more data."""
    fig, ax = plt.subplots(figsize=size)
    ax.plot(n_rows, r2s, marker="o", color=BLUE)
    for x, y_ in zip(n_rows, r2s):
        ax.annotate(f"{y_:.3f}", (x, y_), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    ax.set_xlabel("Training rows used")
    ax.set_ylabel("Test R²")
    ax.set_title("Learning curve (flat end = more data won't help)", fontsize=10)
    ax.set_ylim(min(r2s) - 0.03, max(r2s) + 0.04)
    fig.tight_layout()
    return fig


def plot_group_means(df, col, size=(5.5, 3.2)):
    g = df.groupby(col)[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=size)
    ax.barh(g.index, g.values, color=BLUE)
    ax.axvline(df[TARGET].mean(), color=ORANGE, linestyle="--", label="Overall average")
    ax.set_xlabel("Average productivity rating")
    ax.set_title(f"Productivity by {col.replace('_', ' ')}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_correlations(df, size=(5.5, 3.6)):
    cols = [c for c in NUM_BASE if c != "age"]
    corr = df[cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=size)
    ax.barh(corr.index, corr.values, color=[ORANGE if v < 0 else BLUE for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with productivity")
    ax.set_title("What moves productivity?", fontsize=10)
    fig.tight_layout()
    return fig


SLEEP_BINS = [0, 6, 7, 8, 9, 24]
SLEEP_LABELS = ["<6 h", "6-7 h", "7-8 h", "8-9 h", "9+ h"]
CAFFEINE_BINS = [0, 100, 200, 300, 400, 5000]
CAFFEINE_LABELS = ["<100", "100-200", "200-300", "300-400", "400+"]


def plot_rating_distribution(df, size=(7, 4)):
    counts = df[TARGET].round().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=size)
    ax.bar(counts.index.astype(int), counts.values, color=BLUE)
    ax.axvline(df[TARGET].mean(), color=ORANGE, linestyle="--",
               label=f"Average {df[TARGET].mean():.2f}")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Productivity rating")
    ax.set_ylabel("Number of days")
    ax.set_title("How productive are the days? (rating distribution)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_monthly_trend(df, size=(7, 4)):
    m = df.groupby(df["date"].dt.to_period("M"))[TARGET].mean()
    fig, ax = plt.subplots(figsize=size)
    ax.plot(range(len(m)), m.values, marker="o", markersize=3, color=BLUE)
    ax.axhline(df[TARGET].mean(), color=ORANGE, linestyle="--", label="Overall average")
    step = max(len(m) // 8, 1)
    ax.set_xticks(range(0, len(m), step))
    ax.set_xticklabels([str(p) for p in m.index[::step]], rotation=45, fontsize=7)
    ax.set_ylabel("Average productivity rating")
    ax.set_title("Average productivity by month", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_band_means(df, size=(7, 3.8)):
    """Average productivity for different sleep and caffeine levels."""
    overall = df[TARGET].mean()
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    for ax, (col, bins, labels, title) in zip(axes, [
            ("sleep_hours", SLEEP_BINS, SLEEP_LABELS, "By sleep hours"),
            ("caffeine_mg", CAFFEINE_BINS, CAFFEINE_LABELS, "By caffeine (mg)")]):
        g = df.groupby(pd.cut(df[col], bins, labels=labels, right=False),
                       observed=True)[TARGET].mean()
        ax.bar(g.index.astype(str), g.values, color=BLUE)
        ax.axhline(overall, color=ORANGE, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Average productivity rating")
    axes[0].set_ylim(max(0, min(df[TARGET].mean() - 2, 4)), None)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 4. DOWNLOADABLE REPORTS
# ---------------------------------------------------------------------
def build_pdf_report(df):
    """Productivity report (PDF): overview, highlights and charts. Returns bytes."""
    y = df[TARGET]
    pretty = lambda s: s.replace("_", " ")

    def best_worst(col):
        g = df.groupby(col)[TARGET].mean().sort_values()
        return g.index[-1], g.iloc[-1], g.index[0], g.iloc[0]

    corr = (df[[c for c in NUM_BASE if c != "age"] + [TARGET]]
            .corr()[TARGET].drop(TARGET).sort_values())
    months = df.groupby(df["date"].dt.to_period("M"))[TARGET].mean().sort_values()
    sleep_b = df.groupby(pd.cut(df["sleep_hours"], SLEEP_BINS, labels=SLEEP_LABELS,
                                right=False), observed=True)[TARGET].mean()
    caf_b = df.groupby(pd.cut(df["caffeine_mg"], CAFFEINE_BINS, labels=CAFFEINE_LABELS,
                              right=False), observed=True)[TARGET].mean()
    chb, chbv, chw, chwv = best_worst("chronotype")
    dyb, dybv, dyw, dywv = best_worst("day_of_week")
    ocb, ocbv, ocw, ocwv = best_worst("occupation")

    lines = [
        "OVERVIEW",
        f"  Records: {len(df):,}    Users: {df['user_id'].nunique():,}",
        f"  Period: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
        f"  Average rating: {y.mean():.2f} / 10    Median: {y.median():.0f}",
        f"  High days (8 or more): {(y >= 8).mean():.0%}    Low days (4 or less): {(y <= 4).mean():.0%}",
        "",
        "WHO AND WHEN IS MOST PRODUCTIVE",
        f"  Chronotype: best {chb} ({chbv:.2f}), lowest {chw} ({chwv:.2f})",
        f"  Day of week: best {dyb} ({dybv:.2f}), lowest {dyw} ({dywv:.2f})",
        f"  Occupation: best {ocb} ({ocbv:.2f}), lowest {ocw} ({ocwv:.2f})",
        f"  Month: best {months.index[-1]} ({months.iloc[-1]:.2f}), lowest {months.index[0]} ({months.iloc[0]:.2f})",
        "",
        "WHAT MOVES PRODUCTIVITY (correlation with rating)",
        f"  Strongest positive: {pretty(corr.index[-1])} ({corr.iloc[-1]:+.2f})",
        f"  Second strongest:   {pretty(corr.index[-2])} ({corr.iloc[-2]:+.2f})",
        f"  Strongest negative: {pretty(corr.index[0])} ({corr.iloc[0]:+.2f})",
        "",
        "AVERAGE RATING BY SLEEP AND CAFFEINE",
        "  Sleep:    " + "  ".join(f"{k} {v:.1f}" for k, v in sleep_b.items()),
        "  Caffeine: " + "  ".join(f"{k} {v:.1f}" for k, v in caf_b.items()),
    ]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, "Productivity Report", fontsize=20, fontweight="bold", va="top")
        fig.text(0.08, 0.915, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
                 fontsize=10, va="top", color="#555555")
        yy = 0.875
        for line in lines:
            head = bool(line) and not line.startswith(" ")
            fig.text(0.08, yy, line, fontsize=12 if head else 10,
                     fontweight="bold" if head else "normal", va="top",
                     family=None if head else "monospace")
            yy -= 0.032 if head else 0.027
        pdf.savefig(fig)
        plt.close(fig)

        figs = [
            plot_rating_distribution(df, (8, 4.5)),
            plot_monthly_trend(df, (8, 4.5)),
            plot_group_means(df, "chronotype", (8, 4)),
            plot_group_means(df, "day_of_week", (8, 4.5)),
            plot_group_means(df, "occupation", (8, 5)),
            plot_group_means(df, "weather", (8, 4.5)),
            plot_correlations(df, (8, 5)),
            plot_band_means(df, (8, 4.5)),
        ]
        for f in figs:
            pdf.savefig(f)
            plt.close(f)
    return buf.getvalue()


def build_predictions_csv(res):
    out = res["X_test"].copy()
    out["actual_productivity"] = res["y_test"].values
    out["predicted_productivity"] = np.round(res["pred"], 2)
    out["residual"] = np.round(out["actual_productivity"] - out["predicted_productivity"], 2)
    return out.to_csv(index=False).encode("utf-8")



# ---------------------------------------------------------------------
# 5. DEADLINE RISK
# ---------------------------------------------------------------------
def deadline_risk(task_hours, days_left, hours_per_day, predicted_rating, avg_rating, rmse):
    """Simple, explainable deadline-risk model.

    * A predicted rating equal to the dataset average = a normal day (efficiency 1.0).
      Better days give efficiency > 1, worse days < 1.
    * Useful hours per day = hours you can work x efficiency.
    * The model's typical error (RMSE) tells us how uncertain each day is.
    * Risk % = chance that total useful hours end up below the hours needed.
    """
    eff = max(predicted_rating / avg_rating, 0.2)
    daily_cap = hours_per_day * eff
    cap_mean = daily_cap * days_left
    daily_sd = hours_per_day * rmse / avg_rating
    cap_sd = max(daily_sd * sqrt(days_left), 1e-6)
    z = (task_hours - cap_mean) / cap_sd
    p_miss = 0.5 * (1 + erf(z / sqrt(2)))

    if p_miss < 0.25:
        label, icon = "Low", "🟢"
    elif p_miss < 0.60:
        label, icon = "Medium", "🟡"
    else:
        label, icon = "High", "🔴"

    days_needed = int(np.ceil(task_hours / daily_cap))
    return {
        "p_miss": float(p_miss), "label": label, "icon": icon,
        "efficiency": float(eff), "daily_capacity": float(daily_cap),
        "capacity_total": float(cap_mean), "daily_sd": float(daily_sd),
        "hours_needed_per_day": float(task_hours / (days_left * eff)),
        "days_needed": days_needed, "extra_days": max(days_needed - days_left, 0),
    }


def plot_deadline(risk, task_hours, days_left, size=(6.5, 3.6)):
    d = np.arange(0, days_left + 1)
    mean = risk["daily_capacity"] * d
    sd = risk["daily_sd"] * np.sqrt(d)
    fig, ax = plt.subplots(figsize=size)
    ax.plot(d, mean, color=BLUE, label="Expected useful hours")
    ax.fill_between(d, np.maximum(mean - sd, 0), mean + sd, color=BLUE, alpha=0.2,
                    label="Likely range (±1 std)")
    ax.axhline(task_hours, color=ORANGE, linestyle="--", label=f"Work needed ({task_hours:g} h)")
    ax.set_xlabel("Days from now")
    ax.set_ylabel("Cumulative useful hours")
    ax.set_title(f"Deadline Risk: {risk['label']} ({risk['p_miss']:.0%} chance of missing)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# =====================================================================
# ===== UI =====  (Streamlit page)
# =====================================================================
st.set_page_config(page_title="Productivity Dashboard", page_icon="📊", layout="wide")
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:470px;max-width:520px;}</style>",
    unsafe_allow_html=True,
)


missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error("These files are missing from your GitHub repo: " + ", ".join(missing) +
             ". Run the Colab notebook, unzip app_files.zip and upload every file next to app.py.")
    st.stop()


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_data(show_spinner=False)
def get_user_stats():
    return compute_user_stats(get_data())


@st.cache_resource(show_spinner="Loading model...")
def get_results():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_pdf():
    return build_pdf_report(get_data())


df = get_data()
res = get_results()
user_stats = get_user_stats()

# ------------------------- SIDEBAR -----------------------------------
st.sidebar.title("📊 Model Performance")
st.sidebar.metric("R² score", f"{res['r2']:.3f}")
c1, c2 = st.sidebar.columns(2)
c1.metric("MAE", f"{res['mae']:.2f}")
c2.metric("RMSE", f"{res['rmse']:.2f}")

st.sidebar.divider()
st.sidebar.subheader("Model Diagnostics")

for title, make_fig in [
    ("Actual vs Predicted", lambda: plot_actual_vs_pred(res["y_test"], res["pred"], res["r2"])),
    ("Variable Transformation", lambda: plot_transformations(df)),
    ("Residual Plot", lambda: plot_residuals(res["y_test"], res["pred"])),
    ("Feature Importance", lambda: plot_importance(res["importance"])),
]:
    st.sidebar.markdown(f"**{title}**")
    fig = make_fig()
    st.sidebar.pyplot(fig)
    plt.close(fig)

st.sidebar.divider()
st.sidebar.subheader("Download Reports")
st.sidebar.download_button(
    "Download Productivity Report (PDF)",
    data=get_pdf(),
    file_name="productivity_report.pdf",
    mime="application/pdf",
)
st.sidebar.download_button(
    "Download Predictions (CSV)",
    data=build_predictions_csv(res),
    file_name="productivity_predictions.csv",
    mime="text/csv",
)

# ------------------------- MAIN PAGE ---------------------------------
st.title("📊 Productivity Prediction Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Users", f"{df['user_id'].nunique():,}")
k3.metric("Avg productivity", f"{df[TARGET].mean():.2f} / 10")
k4.metric("Model R²", f"{res['r2']:.3f}")

st.subheader("Productivity insights")
a, b = st.columns(2)
with a:
    fig = plot_correlations(df); st.pyplot(fig); plt.close(fig)
    fig = plot_group_means(df, "day_of_week"); st.pyplot(fig); plt.close(fig)
with b:
    fig = plot_group_means(df, "chronotype"); st.pyplot(fig); plt.close(fig)
    fig = plot_group_means(df, "occupation"); st.pyplot(fig); plt.close(fig)

# ------------------------- PREDICTOR ---------------------------------
st.subheader("Predict productivity for a day")
NEW_USER = "New user (no history)"
who = st.selectbox("Whose day?", [NEW_USER] + sorted(df["user_id"].unique()))
mine = df if who == NEW_USER else df[df["user_id"] == who]
med = mine[NUM_BASE].median().fillna(df[NUM_BASE].median())


def clamp(v, lo, hi):
    return float(min(max(v, lo), hi))


def pick(label, col):
    options = sorted(df[col].unique())
    default = mine[col].mode().iloc[0]
    return st.selectbox(label, options, index=options.index(default), key=f"{col}_{who}")


p1, p2, p3 = st.columns(3)
with p1:
    sleep = st.slider("Sleep hours", 1.5, 12.0, clamp(round(med["sleep_hours"], 1), 1.5, 12.0), 0.1, key=f"sleep_{who}")
    bed = st.slider("Bedtime (24 = midnight, 26 = 2 AM)", 20.0, 30.0, clamp(round(med["bedtime_hr"], 1), 20.0, 30.0), 0.25, key=f"bed_{who}")
    screen = st.slider("Screen time (hours)", 0.5, 16.5, clamp(round(med["screen_time_hours"], 1), 0.5, 16.5), 0.1, key=f"screen_{who}")
    mood = st.slider("Mood score", 1.0, 10.0, clamp(round(med["mood_score"], 1), 1.0, 10.0), 0.1, key=f"mood_{who}")
with p2:
    steps = st.number_input("Steps", 0, 42000, int(clamp(med["steps"], 0, 42000)), 500, key=f"steps_{who}")
    exercise = st.slider("Exercise minutes", 0, 140, int(clamp(med["exercise_minutes"], 0, 140)), 5, key=f"ex_{who}")
    caffeine = st.slider("Caffeine (mg)", 0, 800, int(clamp(med["caffeine_mg"], 0, 800)), 10, key=f"caf_{who}")
    age = st.slider("Age", 18, 80, int(clamp(med["age"], 18, 80)), key=f"age_{who}")
with p3:
    chrono = pick("Chronotype", "chronotype")
    occ = pick("Occupation", "occupation")
    gender = pick("Gender", "gender")
    day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                       "Friday", "Saturday", "Sunday"])
    weather = pick("Weather", "weather")
    climate = pick("Climate zone", "climate_zone")
    temp = st.slider("Temperature (°C)", -15.0, 43.0, clamp(round(med["temperature_c"], 1), -15.0, 43.0), 0.5, key=f"temp_{who}")

row = pd.DataFrame([{
    "user_id": "NEW" if who == NEW_USER else who,
    "age": age, "temperature_c": temp, "sleep_hours": sleep, "screen_time_hours": screen,
    "steps": steps, "exercise_minutes": exercise, "caffeine_mg": caffeine,
    "mood_score": mood, "bedtime_hr": bed, "gender": gender, "occupation": occ,
    "chronotype": chrono, "climate_zone": climate, "weather": weather, "day_of_week": day,
}])
pred_one = float(np.clip(res["model"].predict(make_features(row, user_stats))[0], 1, 10))
st.metric("Predicted productivity rating", f"{pred_one:.1f} / 10")


# ------------------------- DEADLINE RISK -----------------------------
st.subheader("⏰ Deadline Risk")
st.caption("Uses the predicted productivity from the panel above "
           f"({pred_one:.1f} / 10) to estimate whether you will finish on time.")
d1, d2, d3 = st.columns(3)
task_hours = d1.number_input("Hours of work needed", 1.0, 500.0, 20.0, 1.0)
days_left = int(d2.number_input("Days until deadline", 1, 60, 5, 1))
hours_pd = d3.slider("Hours you can work per day", 1.0, 12.0, 6.0, 0.5)

risk = deadline_risk(task_hours, days_left, hours_pd, pred_one,
                     float(df[TARGET].mean()), res["rmse"])

r1, r2c, r3 = st.columns(3)
r1.metric("Risk of missing deadline", f"{risk['icon']} {risk['label']}", f"{risk['p_miss']:.0%}", delta_color="off")
r2c.metric("Useful hours per day", f"{risk['daily_capacity']:.1f} h",
           f"{risk['efficiency']:.0%} of a normal day", delta_color="off")
r3.metric("Hours/day needed", f"{risk['hours_needed_per_day']:.1f} h")
st.progress(min(max(risk["p_miss"], 0.0), 1.0))

if risk["label"] == "High":
    st.error(f"At this productivity level you would need about {risk['days_needed']} days "
             f"({risk['extra_days']} more than you have), or {risk['hours_needed_per_day']:.1f} "
             "working hours per day.")
elif risk["label"] == "Medium":
    st.warning("It is tight. A low-productivity day (poor sleep, lots of caffeine or screen time) "
               "could push you past the deadline. Try to build in a buffer day.")
else:
    st.success("You are on track, even allowing for some bad days.")

fig = plot_deadline(risk, task_hours, days_left)
st.pyplot(fig)
plt.close(fig)
