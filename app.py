from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
METRICS_PATH = ROOT / "results" / "model_metrics.csv"
IMPORTANCE_PATH = ROOT / "results" / "feature_importance.csv"

st.set_page_config(
    page_title="Student Placement Readiness Predictor",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


st.title("🎓 Student Placement / Internship Readiness Predictor")
st.write(
    "Enter student academic and skill details to estimate placement readiness using "
    "the best-performing supervised machine learning model."
)

st.info(
    "This is an educational prediction tool trained on a synthetic Kaggle dataset. "
    "It should be used for learning and self-assessment, not for real hiring, admission, "
    "or high-stakes decisions."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("The trained model has not been generated yet.")
    st.markdown("Run these commands from the project folder:")
    st.code(
        "python src/download_data.py\n"
        "python src/train.py\n"
        "streamlit run app.py",
        language="bash",
    )
    st.stop()

model = load_model()
metadata = load_metadata()

left, right = st.columns([2, 1])
with right:
    st.subheader("Model information")
    st.write(f"**Best model:** {metadata['best_model']}")
    st.write(f"**Target:** {metadata['target_column']}")
    st.write(f"**Selection metric:** {metadata['selection_metric']}")
    st.write(f"**Dataset file:** {metadata['dataset_file']}")

with left:
    st.subheader("Student details")

    inputs: dict[str, object] = {}
    with st.form("prediction_form"):
        feature_schema = metadata["feature_schema"]
        form_columns = st.columns(2)

        for index, feature in enumerate(feature_schema):
            container = form_columns[index % 2]
            name = feature["name"]

            with container:
                if feature["type"] == "numeric":
                    minimum = float(feature.get("min", 0.0))
                    maximum = float(feature.get("max", 100.0))
                    default = float(feature.get("default", minimum))
                    span = maximum - minimum
                    step = 1.0 if span >= 20 and default.is_integer() else 0.1
                    inputs[name] = st.number_input(
                        name,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=step,
                    )
                else:
                    choices = feature.get("choices") or [feature.get("default", "Unknown")]
                    default = feature.get("default", choices[0])
                    default_index = choices.index(default) if default in choices else 0
                    inputs[name] = st.selectbox(name, choices, index=default_index)

        submitted = st.form_submit_button("Predict readiness", type="primary")

if submitted:
    input_df = pd.DataFrame([inputs], columns=metadata["feature_columns"])
    prediction = int(model.predict(input_df)[0])
    label = metadata["target_mapping"].get(str(prediction), str(prediction))

    st.divider()
    st.subheader("Prediction")

    if prediction == 1:
        st.success(f"Predicted status: **{label}**")
    else:
        st.warning(f"Predicted status: **{label}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_positions = list(model.classes_)
        if 1 in class_positions:
            positive_probability = float(probabilities[class_positions.index(1)])
            st.metric(
                "Estimated placement-readiness probability",
                f"{positive_probability * 100:.1f}%",
            )

    st.caption(
        "Treat this result as a learning signal. Improve weak academic, technical, project, "
        "internship, aptitude, and communication areas rather than treating the prediction "
        "as a final judgement."
    )

st.divider()

if METRICS_PATH.exists():
    st.subheader("Model comparison")
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    chart_columns = [column for column in ["accuracy", "f1_score"] if column in metrics.columns]
    if chart_columns:
        st.bar_chart(metrics.set_index("model")[chart_columns])

if IMPORTANCE_PATH.exists():
    st.subheader("Important factors")
    importance = pd.read_csv(IMPORTANCE_PATH).head(10)
    st.write(
        "These are the input features that most affected prediction performance for the "
        "selected model, estimated using permutation importance."
    )
    st.dataframe(importance, use_container_width=True, hide_index=True)
    if not importance.empty:
        st.bar_chart(importance.set_index("feature")["importance_mean"])
