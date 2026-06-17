import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import joblib
from ml.predict import predict_failure

st.set_page_config(
    page_title="AI Predictive Maintenance",
    layout="wide"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🤖 AI Assistant",
        "📊 Dataset Explorer",
        "🔮 Failure Prediction",
        "📚 Project Explanation"
    ]
)

if page == "🤖 AI Assistant":

    st.title("Maintenance Assistant")

    question = st.text_area(
        "Ask a maintenance question"
    )

    if st.button("Ask"):

        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={
                "question": question,
                "machine_id": "1"
            }
        )

        st.write(response.json()["answer"])


if page == "📊 Dataset Explorer":

    df = pd.read_csv(
        "datasets/ai4i2020.csv"
    )

    st.dataframe(df.head())

    st.subheader("Dataset statistics")

    st.dataframe(
        df.describe()
    )

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True
    )

    st.plotly_chart(fig)
    
    fig = px.histogram(
    df,
    x="Machine failure"
    )

    st.plotly_chart(fig)

if page == "🔮 Failure Prediction":

    st.title("🔮 Failure Prediction")

    air_temp = st.slider(
        "Air Temperature [K]",
        290,
        320,
        300
    )

    process_temp = st.slider(
        "Process Temperature [K]",
        300,
        340,
        310
    )

    rpm = st.slider(
        "Rotational Speed [rpm]",
        1000,
        3000,
        1500
    )

    torque = st.slider(
        "Torque [Nm]",
        0,
        100,
        40
    )

    wear = st.slider(
        "Tool Wear [min]",
        0,
        300,
        50
    )

    if st.button("Predict Failure"):

        prediction, probability = predict_failure(
            air_temp,
            process_temp,
            rpm,
            torque,
            wear
        )

        st.metric(
            "Failure Probability",
            f"{probability:.1%}"
        )

        if prediction == 1:

            st.error(
                f"⚠️ Failure risk detected ({probability:.1%})"
            )

        else:

            st.success(
                f"✅ Machine healthy ({1-probability:.1%})"
            )

if page == "📚 Project Explanation":

    st.title("📚 Project Overview")

    st.markdown("""
    ## AI-Powered Predictive Maintenance

    This project combines:

    - Industrial data analysis
    - Machine Learning prediction
    - Retrieval Augmented Generation (RAG)
    - Large Language Models (TinyLlama)

    Workflow:

    Dataset → Machine Learning → Failure Prediction

    PDF Manual → Vector Database → RAG Assistant

    ML + RAG → Intelligent Maintenance Agent
    """)