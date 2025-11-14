import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="🌤️ Weather Temperature Prediction", layout="wide")
st.title("🌦️ Predict Temperature")

# Load trained model
try:
    model = joblib.load("models/best_weather_regressor.pkl")
except:
    st.error("❌ Model not found! Please run `train_model.py` first to train and save the model.")
    st.stop()

# Collect inputs
st.subheader("Enter Weather Details:")

humidity = st.number_input("Enter humidity:", min_value=0.0, max_value=100.0, value=64.57)
wind_speed = st.number_input("Enter wind_speed (km/h):", min_value=0.0, value=19.92)
pressure = st.number_input("Enter pressure (hPa):", min_value=900.0, max_value=1100.0, value=1013.23)
visibility = st.number_input("Enter visibility (km):", min_value=0.0, value=5.53)
dew_point = st.number_input("Enter dew_point (°C):", value=20.12)
cloud_cover = st.number_input("Enter cloud_cover (%):", min_value=0.0, max_value=100.0, value=48.70)
region = st.selectbox("Select region:", ["North", "South", "East", "West"])
season = st.selectbox("Select season:", ["Summer", "Monsoon", "Winter", "Autumn"])
weather_condition = st.selectbox("Select weather_condition:", ["Sunny", "Cloudy", "Rainy", "Stormy"])

# Prepare input data
input_df = pd.DataFrame([{
    "humidity": humidity,
    "wind_speed": wind_speed,
    "pressure": pressure,
    "visibility": visibility,
    "dew_point": dew_point,
    "cloud_cover": cloud_cover,
    "region": region,
    "season": season,
    "weather_condition": weather_condition
}])

# Prediction button
if st.button("Predict Temperature"):
    prediction = model.predict(input_df)
    st.success(f"🌈 Predicted temperature: **{prediction[0]:.2f} °C**")
