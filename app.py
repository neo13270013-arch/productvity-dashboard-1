
import io, json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

st.set_page_config(page_title="Productivity & Deadline Dashboard", page_icon="📊", layout="wide")
BASE = Path(__file__).parent
MODEL_PATH = BASE / "productivity_model.joblib"
META_PATH = BASE / "productivity_metadata.json"
DATA_PATH = BASE / "everyday_life_dataset.csv"

st.title("📊 Productivity & Deadline Risk Dashboard")
st.caption("Daily productivity estimates, model diagnostics, and deadline planning.")

@st.cache_resource
def load_assets():
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, meta

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Place productivity_model.joblib and productivity_metadata.json in the same folder as app.py.")
    st.stop()

try:
    model, meta = load_assets()
except Exception as e:
    st.error(f"Could not load model: {e}. Ensure requirements.txt matches the model's scikit-learn version.")
    st.stop()

threshold = float(meta.get("low_productivity_threshold", 5.0))
data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None

def transform_inputs(frame):
    x = frame.copy()
    bt = pd.to_datetime(x["bedtime"], format="%H:%M", errors="coerce")
    mins = bt.dt.hour * 60 + bt.dt.minute
    x["bedtime_minutes"] = mins
    x["bedtime_sin"] = np.sin(2 * np.pi * mins / 1440)
    x["bedtime_cos"] = np.cos(2 * np.pi * mins / 1440)
    x = x.drop(columns=["bedtime", "date"], errors="ignore")
    # Keep raw screen time as well as its transformed version; the trained model uses both.
    for col in ["steps", "exercise_minutes", "caffeine_mg", "screen_time_hours"]:
        x[col + "_log1p"] = np.log1p(pd.to_numeric(x[col], errors="coerce").clip(lower=0))
        if col != "screen_time_hours":
            x = x.drop(columns=[col], errors="ignore")
    return x[meta["model_input_columns"]]

st.sidebar.header("Model performance")
st.sidebar.metric("R²", f"{meta['r2']:.3f}")
st.sidebar.metric("MAE", f"{meta['mae']:.3f}")
st.sidebar.metric("RMSE", f"{meta['rmse']:.3f}")
st.sidebar.write(f"Low-productivity threshold: {threshold:.1f}")
st.sidebar.caption("Metrics use a held-out 20% test set.")

st.header("1. Predict daily productivity")
with st.form("prediction_form"):
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(meta["feature_columns"]):
        spec = meta["features"][feature]
        with (left if i % 2 == 0 else right):
            label = feature.replace("_", " ").title()
            if spec["type"] == "numeric":
                values[feature] = st.number_input(
                    label, value=float(spec["median"]),
                    min_value=float(spec.get("min", -1e9)),
                    max_value=float(spec.get("max", 1e9))
                )
            else:
                choices = spec.get("choices", ["Unknown"])
                default = spec.get("default", choices[0])
                if default not in choices: default = choices[0]
                values[feature] = st.selectbox(label, choices, index=choices.index(default))
    st.divider()
    st.subheader("Deadline information")
    days_left = st.number_input("Days remaining until deadline", 0, 3650, 3)
    tasks_left = st.number_input("Tasks remaining", 0, 10000, 3)
    submitted = st.form_submit_button("Predict productivity", type="primary")

if "report_result" not in st.session_state:
    st.session_state.report_result = None

if submitted:
    try:
        raw = pd.DataFrame([values], columns=meta["feature_columns"])
        pred = float(model.predict(transform_inputs(raw))[0])
        shown = float(np.clip(pred, meta["target_min"], meta["target_max"]))
        if tasks_left == 0:
            risk, reason = "Low", "No tasks remain unfinished."
        elif days_left == 0:
            risk, reason = "High", "Tasks remain and the deadline is today."
        elif pred <= threshold and days_left <= 3:
            risk, reason = "High", "Low predicted productivity and a near deadline coincide with unfinished tasks."
        elif days_left <= 1:
            risk, reason = "High", "The deadline is very close and tasks remain."
        elif pred <= threshold or days_left <= 3:
            risk, reason = "Medium", "A productivity warning or short deadline suggests closer monitoring."
        else:
            risk, reason = "Low", "Configured warning conditions are not met; this does not guarantee on-time completion."

        st.header("2. Prediction results")
        a, b = st.columns(2)
        a.metric("Predicted productivity", f"{shown:.2f} / 10")
        b.metric("Low-productivity threshold", f"{threshold:.2f}")
        if pred <= threshold: st.warning("Low-productivity warning: predicted score is at or below the threshold.")
        else: st.success("Predicted score is above the low-productivity threshold.")

        st.header("3. Deadline risk assessment")
        if risk == "High": st.error(f"🔴 HIGH — {reason}")
        elif risk == "Medium": st.warning(f"🟠 MEDIUM — {reason}")
        else: st.success(f"🟢 LOW — {reason}")
        st.caption("Deadline risk is a transparent rule-based estimate, not a trained deadline-outcome prediction.")
        st.session_state.report_result = {
            "prediction": shown, "risk": risk, "reason": reason,
            "days_left": int(days_left), "tasks_left": int(tasks_left),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.header("4. Model evaluation graphs")
st.caption("All evaluation plots below use the held-out test data.")
actual = np.asarray(meta["diagnostics"]["actual"])
predicted = np.asarray(meta["diagnostics"]["predicted"])
resid = actual - predicted
fig1, ax = plt.subplots(figsize=(6, 4))
ax.scatter(actual, predicted, alpha=.35, s=12)
low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
ax.plot([low, high], [low, high], "--")
ax.set(xlabel="Actual productivity rating", ylabel="Predicted productivity rating", title="Actual vs Predicted")
ax.grid(alpha=.2)
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

fig2, ax = plt.subplots(figsize=(6, 4))
ax.scatter(predicted, resid, alpha=.35, s=12)
ax.axhline(0, linestyle="--")
ax.set(xlabel="Predicted productivity rating", ylabel="Residual (actual − predicted)", title="Residual Plot")
ax.grid(alpha=.2)
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

imp = pd.DataFrame(meta["feature_importance"]).head(15).sort_values("importance")
fig3, ax = plt.subplots(figsize=(7, 5))
ax.barh(imp["feature"], imp["importance"])
ax.set(title="Top 15 Feature Importances", xlabel="Importance")
ax.grid(axis="x", alpha=.2)
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

st.subheader("Variable transformation graph")
st.write("log1p reduces the scale of highly skewed numeric variables. Bedtime is encoded cyclically to handle times around midnight.")
if data is not None:
    chosen = st.selectbox("Select a variable", ["steps", "exercise_minutes", "caffeine_mg", "screen_time_hours"])
    vals = pd.to_numeric(data[chosen], errors="coerce").dropna().clip(lower=0)
    fig4, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(vals, bins=30)
    axes[0].set_title("Before transformation")
    axes[0].set_xlabel(chosen)
    axes[1].hist(np.log1p(vals), bins=30)
    axes[1].set_title("After log1p")
    axes[1].set_xlabel("log(1 + value)")
    for axis in axes: axis.set_ylabel("Frequency"); axis.grid(alpha=.15)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

st.header("5. Download PDF report")
st.write("The PDF includes model metrics, the four diagnostic visualizations, and the latest prediction if one has been made.")

def chart_png(kind):
    buff = io.BytesIO()
    if kind == "actual":
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(actual, predicted, alpha=.35, s=12)
        ax.plot([low, high], [low, high], "--")
        ax.set(xlabel="Actual rating", ylabel="Predicted rating", title="Actual vs Predicted")
    elif kind == "residual":
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(predicted, resid, alpha=.35, s=12); ax.axhline(0, linestyle="--")
        ax.set(xlabel="Predicted rating", ylabel="Residual", title="Residual Plot")
    elif kind == "importance":
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(imp["feature"], imp["importance"])
        ax.set(title="Top Feature Importances", xlabel="Importance")
    else:
        vals = pd.to_numeric(data[kind], errors="coerce").dropna().clip(lower=0)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].hist(vals, bins=25); axes[0].set_title("Before")
        axes[1].hist(np.log1p(vals), bins=25); axes[1].set_title("After log1p")
    fig.savefig(buff, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig); buff.seek(0)
    return buff

def create_pdf():
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Productivity Analysis Report", styles["Title"]),
        Paragraph("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]),
        Spacer(1, 12), Paragraph("Model performance", styles["Heading2"])
    ]
    table = Table([
        ["Metric", "Value"],
        ["R²", f"{meta['r2']:.3f}"],
        ["MAE", f"{meta['mae']:.3f}"],
        ["RMSE", f"{meta['rmse']:.3f}"]
    ], colWidths=[220, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("PADDING", (0,0), (-1,-1), 7)
    ]))
    story += [table, Spacer(1, 12)]
    result = st.session_state.report_result
    if result:
        story += [
            Paragraph("Latest prediction", styles["Heading2"]),
            Paragraph(f"Predicted productivity: <b>{result['prediction']:.2f} / 10</b>", styles["Normal"]),
            Paragraph(f"Deadline risk: <b>{result['risk']}</b>", styles["Normal"]),
            Paragraph(result["reason"], styles["Normal"]),
            Paragraph(f"Days remaining: {result['days_left']} | Tasks remaining: {result['tasks_left']}", styles["Normal"]),
            Spacer(1, 10)
        ]
    else:
        story += [Paragraph("No prediction was submitted in this session. This report contains model evaluation results only.", styles["Normal"]), Spacer(1, 10)]
    story.append(Paragraph("Actual vs Predicted", styles["Heading2"]))
    story.append(Image(chart_png("actual"), width=460, height=260))
    story.append(Paragraph("Residual Plot", styles["Heading2"]))
    story.append(Image(chart_png("residual"), width=460, height=260))
    story.append(Paragraph("Feature Importance", styles["Heading2"]))
    story.append(Image(chart_png("importance"), width=460, height=260))
    if data is not None:
        story.append(Paragraph("Variable Transformation", styles["Heading2"]))
        story.append(Image(chart_png("steps"), width=460, height=180))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Note: Predictions are estimates. Deadline risk is based on simple rules and is not a trained deadline-outcome model.", styles["Italic"]))
    doc.build(story); buff.seek(0)
    return buff.getvalue()

if st.button("Prepare PDF report", type="primary"):
    try:
        st.session_state.pdf_ready = create_pdf()
    except Exception as e:
        st.error(f"Could not create PDF: {e}")
if st.session_state.get("pdf_ready"):
    st.download_button("⬇️ Download PDF report", st.session_state.pdf_ready,
                       file_name="productivity_analysis_report.pdf", mime="application/pdf")

st.header("6. Dataset overview")
if data is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(data):,}")
    c2.metric("Dataset columns", len(data.columns))
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    with st.expander("Preview dataset"):
        st.dataframe(data.head(20), use_container_width=True)
    st.metric("Average productivity rating", f"{data[meta['target_column']].mean():.2f} / 10")

st.divider()
st.caption("Prototype for productivity planning. Predictions are estimates, not definitive judgments about a person.")
