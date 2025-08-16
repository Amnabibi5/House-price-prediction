import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Available models and their filenames
model_options = {
    "Linear Regression": "LinearRegression.pkl",
    "Ridge Regression": "Ridge.pkl",
    "Random Forest": "RandomForestRegressor.pkl"
}

# Precomputed performance metrics (example values)
model_metrics = {
    "Linear Regression": {"RMSE": 1200000, "R²": 0.82},
    "Ridge Regression": {"RMSE": 1180000, "R²": 0.83},
    "Random Forest": {"RMSE": 950000, "R²": 0.89}
}

# Load label encoder
encoder_path = "models/label_encoder.pkl"
if not os.path.exists(encoder_path):
    st.error("❌ Label encoder file not found.")
    st.stop()

label_encoder = joblib.load(encoder_path)

st.title("🏠 House Price Prediction Dashboard")

with st.expander("ℹ️ Input Guide"):
    st.markdown("""
    Fill in the house details below. The app will predict prices using multiple models and show performance metrics.
    """)

# Input form
st.header("📋 House Details")

area = st.number_input("📐 Area (sq ft)", min_value=500, max_value=10000, step=50)
bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4])
stories = st.selectbox("🏢 Stories", [1, 2, 3])
parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3])
guestroom = st.selectbox("🛋 Guest Room", ["Yes", "No"])
basement = st.selectbox("🏚 Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("❄️ Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("🌟 Preferred Area", ["Yes", "No"])
furnishingstatus = st.selectbox("🪑 Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
mainroad = st.selectbox("🛣 Main Road Access", ["Yes", "No"])
location = st.selectbox("📍 Location", label_encoder.classes_)

# Predict button
if st.button("🔍 Compare Predictions"):
    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "location": location,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
        "mainroad": mainroad
    }

    input_df = pd.DataFrame([input_dict])

    # Map binary features
    binary_map = {"Yes": 1, "No": 0}
    for col in ["guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "mainroad"]:
        input_df[col] = input_df[col].map(binary_map)

    # Encode location
    try:
        input_df["location"] = label_encoder.transform(input_df["location"])
    except Exception as e:
        st.error(f"❌ Location encoding failed: {e}")
        st.stop()

    # Predict with all models
    predictions = {}
    for model_name, filename in model_options.items():
        model_path = f"models/{filename}"
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model '{model_name}' not found.")
            continue
        try:
            model = joblib.load(model_path)
            price = model.predict(input_df)[0]
            predictions[model_name] = price
        except Exception as e:
            st.warning(f"⚠️ Prediction failed for '{model_name}': {e}")

    # Show predictions
    st.subheader("💰 Predicted Prices")
    for model_name, price in predictions.items():
        st.write(f"**{model_name}**: PKR {price:,.0f}")

    # Show chart
    if predictions:
        st.subheader("📊 Price Comparison Chart")
        fig, ax = plt.subplots()
        ax.bar(predictions.keys(), predictions.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_ylabel("Price (PKR)")
        ax.set_title("Predicted House Prices by Model")
        st.pyplot(fig)

    # Show metrics
    st.subheader("📐 Model Performance Metrics")
    metrics_df = pd.DataFrame(model_metrics).T
    st.dataframe(metrics_df.style.format({"RMSE": "{:,.0f}", "R²": "{:.2f}"}))

