from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Graduate Admission Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Graduate Admission Prediction Dashboard")
st.write(
    "Interactive university admission prediction dashboard based on "
    "SageMaker Canvas batch prediction results."
)

uploaded_file = st.file_uploader(
    "Upload another SageMaker prediction CSV (optional)",
    type=["csv"]
)

default_file = Path("canvas_predictions.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif default_file.exists():
    df = pd.read_csv(default_file)
else:
    st.info("Please upload the SageMaker prediction CSV to start.")
    st.stop()

required_columns = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
    "Chance of Admit"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

# Prepare regression data from the Canvas prediction results
features = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research"
]

X = df[features].astype(float).values
y = df["Chance of Admit"].astype(float).values

# Add intercept and calculate regression coefficients
X_design = np.column_stack([np.ones(len(X)), X])
coefficients = np.linalg.lstsq(X_design, y, rcond=None)[0]

st.sidebar.header("Student Inputs")

gre = st.sidebar.slider(
    "GRE Score",
    int(df["GRE Score"].min()),
    int(df["GRE Score"].max()),
    int(df["GRE Score"].mean())
)

toefl = st.sidebar.slider(
    "TOEFL Score",
    int(df["TOEFL Score"].min()),
    int(df["TOEFL Score"].max()),
    int(df["TOEFL Score"].mean())
)

university_rating = st.sidebar.selectbox(
    "University Rating",
    sorted(df["University Rating"].unique()),
    index=2 if len(df["University Rating"].unique()) >= 3 else 0
)

sop = st.sidebar.slider(
    "SOP",
    float(df["SOP"].min()),
    float(df["SOP"].max()),
    float(df["SOP"].mean()),
    0.5
)

lor = st.sidebar.slider(
    "LOR",
    float(df["LOR"].min()),
    float(df["LOR"].max()),
    float(df["LOR"].mean()),
    0.5
)

cgpa = st.sidebar.slider(
    "CGPA",
    float(df["CGPA"].min()),
    float(df["CGPA"].max()),
    float(df["CGPA"].mean()),
    0.01
)

research = st.sidebar.selectbox(
    "Research Experience",
    sorted(df["Research"].unique())
)

input_values = np.array([
    gre,
    toefl,
    university_rating,
    sop,
    lor,
    cgpa,
    research
], dtype=float)

prediction = coefficients[0] + np.dot(coefficients[1:], input_values)

prediction = float(np.clip(prediction, 0, 1))

col1, col2, col3 = st.columns(3)

col1.metric("Estimated Chance of Admit", f"{prediction:.1%}")
col2.metric("CGPA", f"{cgpa:.2f}")
col3.metric("GRE Score", f"{gre}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Prediction")

    prediction_chart = pd.DataFrame(
        {
            "Admission Chance": [prediction]
        },
        index=["Student"]
    )

st.progress(prediction, text=f"Predicted Chance of Admit: {prediction:.1%}")

with right:
    st.subheader("Selected Student Inputs")

    input_table = pd.DataFrame(
        {
            "Feature": features,
            "Value": input_values
        }
    )

    st.dataframe(
        input_table,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.subheader("Canvas Batch Prediction Results")

st.dataframe(
    df.sort_values(
        "Chance of Admit",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)
