import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="🔥 Fire Type Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(to right, #fdfbfb, #ebedee);
    }

    .block-container {
        padding: 2rem 3rem;
        max-width: 700px;
        margin: auto;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.07);
    }

    h1 {
        text-align: center;
        color: #e63946;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .stMarkdown h2 {
        color: #1d3557;
    }

    .stButton > button {
        background-color: #1d3557;
        color: #fff;
        font-size: 1.1rem;
        padding: 0.6rem 1.4rem;
        border-radius: 8px;
        border: none;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        background-color: #457b9d;
        transition: 0.3s ease;
    }

    .result-box {
        background-color: #dff0d8;
        border-left: 6px solid #3c763d;
        padding: 1rem;
        margin-top: 1.5rem;
        border-radius: 8px;
        font-size: 1.1rem;
    }

    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🔥 Fire Type Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict the type of fire using <b>MODIS satellite</b> input readings 🌍</p>", unsafe_allow_html=True)

# Input fields
brightness = st.number_input("💡 Brightness", value=300.0)
bright_t31 = st.number_input("🌡️ Brightness T31", value=290.0)
frp = st.number_input("🔥 Fire Radiative Power (FRP)", value=15.0)
scan = st.number_input("🛰️ Scan", value=1.0)
track = st.number_input("📍 Track", value=1.0)
confidence = st.selectbox("📶 Confidence Level", ["low", "nominal", "high"],
                          help="Select the confidence level reported by the satellite")

# Convert confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Predict and display
if st.button("🚀 Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "🌿 Vegetation Fire",
        2: "🏞️ Other Static Land Source",
        3: "🌊 Offshore Fire"
    }

    result = fire_types.get(prediction, "❓ Unknown")
    st.markdown(f"<div class='result-box'>✅ <strong>Predicted Fire Type:</strong> {result}</div>", unsafe_allow_html=True)
