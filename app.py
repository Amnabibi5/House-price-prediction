import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and preprocessing pipeline
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# Title and description
st.title("🏠 House Price Prediction Dashboard")
st.markdown("Enter house details below to predict the price. Model is trained on housing data with reproducible ML workflow.")

# Sidebar inputs
st.sidebar.header("📋 Input Features")

def user_input_features():
    data = {
        "area": st.sidebar.slider("Area (sq ft)", 500, 10000, 2000),
        "bedrooms": st.sidebar.slider("Bedrooms", 1, 10, 3),
        "bathrooms": st.sidebar.slider("Bathrooms", 1, 10, 2),
        "stories": st.sidebar.slider("Stories", 1, 4, 2),
        "mainroad": st.sidebar.selectbox("Main Road Access", ["yes", "no"]),
        "guestroom": st.sidebar.selectbox("Guest Room", ["yes", "no"]),
        "basement": st.sidebar.selectbox("Basement", ["yes", "no"]),
        "hotwaterheating": st.sidebar.selectbox("Hot Water Heating", ["yes", "no"]),
        "airconditioning": st.sidebar.selectbox("Air Conditioning", ["yes", "no"]),
        "parking": st.sidebar.slider("Parking Spaces", 0, 5, 1),
        "prefarea": st.sidebar.selectbox("Preferred Area", ["yes", "no"]),
        "furnishingstatus": st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Preprocessing
def preprocess_input(df, feature_cols):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Align with training columns
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_cols]  # Drop any extra columns

    return df_encoded

try:
    input_processed = preprocess_input(input_df, feature_cols)
    input_scaled = preprocessor.transform(input_processed)
    prediction = model.predict(input_scaled)

    st.subheader("💰 Predicted House Price")
    st.success(f"PKR {prediction[0]:,.0f}")

    st.markdown("### 🔍 Input Summary")
    st.dataframe(input_df)

except Exception as e:
    st.error("⚠️ Prediction failed due to input mismatch or preprocessing error.")
    st.code(str(e), language="python")




