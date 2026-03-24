import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.title("AI-Driven Disease Outbreak Prediction (Ghana)")

st.sidebar.header("Input Parameters")
district = st.sidebar.selectbox("District", ["Accra", "Kumasi", "Tamale", "Cape Coast", "Takoradi"])
week_of_year = st.sidebar.slider("Week of Year", 1, 52, 25)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 15.0)
temperature = st.sidebar.slider("Temperature (°C)", 15.0, 40.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 30.0, 90.0, 70.0)
sanitation = st.sidebar.slider("Sanitation Score", 0.0, 1.0, 0.8)
population_density = st.sidebar.slider("Population Density", 100, 1000, 500)
previous_cases = st.sidebar.slider("Previous Cases", 0, 100, 40)

if st.sidebar.button("Predict Outbreak Risk"):
    payload = {
        "district": district,
        "week_of_year": week_of_year,
        "rainfall_mm": rainfall,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "sanitation_score": sanitation,
        "population_density": population_density,
        "previous_cases": previous_cases
    }

    try:
        response = requests.post("http://localhost:5000/predict", json=payload)
        result = response.json()["predictions"][0]

        st.header("Prediction Results")
        risk_level = "High Risk" if result["prediction"] == 1 else "Low Risk"
        st.metric("Outbreak Risk", risk_level)
        st.metric("Confidence", f"{result['confidence']:.2%}")

        # Visualization
        fig, ax = plt.subplots()
        factors = ["Rainfall", "Temperature", "Humidity", "Sanitation", "Population Density", "Previous Cases"]
        values = [rainfall, temperature, humidity, sanitation*100, population_density/10, previous_cases]
        ax.barh(factors, values)
        ax.set_xlabel("Values")
        ax.set_title("Input Factors")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error connecting to API: {e}")

st.header("About")
st.write("""
This dashboard uses a machine learning model to predict disease outbreak risks in Ghana based on:
- Meteorological data (rainfall, temperature, humidity)
- Health data (previous cases)
- Environmental factors (sanitation, population density)

The model is trained on historical data and provides real-time risk assessments.
""")