import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Graduate Admission Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Graduate Admission Prediction Dashboard")
st.write("Interactive dashboard based on SageMaker Canvas batch prediction results.")

uploaded_file = st.file_uploader(
    "Upload the SageMaker prediction CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the batch prediction CSV to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

prediction_col = "Chance of Admit"

st.sidebar.header("Filters")

gre_min = int(df["GRE Score"].min())
gre_max = int(df["GRE Score"].max())

gre_range = st.sidebar.slider(
    "GRE Score",
    gre_min,
    gre_max,
    (gre_min, gre_max)
)

cgpa_min = float(df["CGPA"].min())
cgpa_max = float(df["CGPA"].max())

cgpa_range = st.sidebar.slider(
    "CGPA",
    cgpa_min,
    cgpa_max,
    (cgpa_min, cgpa_max)
)

research_options = sorted(df["Research"].unique())

research_filter = st.sidebar.multiselect(
    "Research",
    research_options,
    default=research_options
)

filtered_df = df[
    df["GRE Score"].between(gre_range[0], gre_range[1])
    & df["CGPA"].between(cgpa_range[0], cgpa_range[1])
    & df["Research"].isin(research_filter)
]

if filtered_df.empty:
    st.warning("No applicants match the selected filters.")
    st.stop()

avg_prediction = filtered_df[prediction_col].mean()
high_chance = (filtered_df[prediction_col] >= 0.80).sum()
avg_cgpa = filtered_df["CGPA"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Applicants", len(filtered_df))
col2.metric("Average Predicted Chance", f"{avg_prediction:.1%}")
col3.metric("Chance ≥ 80%", high_chance)
col4.metric("Average CGPA", f"{avg_cgpa:.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Predicted Admission Chance")

    chart_data = filtered_df[[prediction_col]].copy()

    chart_data["Range"] = pd.cut(
        chart_data[prediction_col],
        bins=10
    )

    counts = (
        chart_data["Range"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    counts.columns = ["Range", "Applicants"]

    counts["Range"] = counts["Range"].astype(str)

    st.bar_chart(
        counts.set_index("Range")["Applicants"]
    )

with right:
    st.subheader("CGPA vs Predicted Chance")

    scatter_data = filtered_df[
        ["CGPA", prediction_col]
    ].rename(
        columns={prediction_col: "Predicted Chance"}
    )

    st.scatter_chart(
        scatter_data,
        x="CGPA",
        y="Predicted Chance"
    )

st.subheader("Prediction Results")

display_columns = [
    "Serial No.",
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
    "Chance of Admit"
]

st.dataframe(
    filtered_df[display_columns].sort_values(
        "Chance of Admit",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)
