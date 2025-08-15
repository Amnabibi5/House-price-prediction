import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Load label encoder
try:
    le = joblib.load("models/label_encoder.pkl")
except:
    le = None
    st.warning("⚠️ Label encoder not found. Classification labels may not display correctly.")

# Model options
model_names = [
    "LinearRegression", "Ridge", "Lasso", "RandomForestRegressor", "GradientBoostingRegressor",
    "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "SVC"
]

def is_regression(model_name):
    return model_name.endswith("Regressor") or model_name in ["LinearRegression", "Ridge", "Lasso"]

# Sidebar
st.sidebar.title("🔍 Model Selection")
model_descriptions = {
    "LinearRegression": "Simple linear model for price prediction",
    "Ridge": "Linear model with L2 regularization",
    "Lasso": "Linear model with L1 regularization",
    "RandomForestRegressor": "Ensemble of decision trees (regression)",
    "GradientBoostingRegressor": "Boosted trees for better accuracy",
    "LogisticRegression": "Simple classifier for price category",
    "RandomForestClassifier": "Ensemble classifier for categories",
    "GradientBoostingClassifier": "Boosted classifier for categories",
    "SVC": "Support Vector Classifier"
}

selected_model = st.sidebar.selectbox(
    "🔍 Choose a model",
    model_names,
    format_func=lambda name: f"{name} — {model_descriptions.get(name, '')}"
)

# Load model
model_path = f"models/{selected_model}.pkl"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{selected_model}.pkl' not found in 'models/' folder.")
    st.stop()

model = joblib.load(model_path)

# Input form
st.title("🏠 House Price Prediction Dashboard")
st.write("Enter house specifications below:")

with st.form("prediction_form"):
    area = st.number_input("Area (sq ft)", min_value=100)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    stories = st.slider("Stories", 1, 3, 1)
    parking = st.slider("Parking Spaces", 0, 3, 1)
    guestroom = st.selectbox("Guest Room", ["Yes", "No"])
    basement = st.selectbox("Basement", ["Yes", "No"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
    airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
    prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
    furnishingstatus = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
    submitted = st.form_submit_button("Predict")

# Prepare input
if submitted:
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
        "furnishingstatus": furnishingstatus
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    binary_map = {"Yes": 1, "No": 0}
    for col in ["guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        input_df[col] = input_df[col].map(binary_map)

    if le:
        try:
            input_df["location"] = le.transform(input_df["location"])
        except ValueError:
            st.warning("⚠️ Location encoding failed. Check if label encoder matches input categories.")
            st.stop()
    else:
        st.warning("⚠️ Location not encoded. Model may not perform correctly.")

    # Prediction
    try:
        prediction = model.predict(input_df)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    if is_regression(selected_model):
        price = prediction[0]
        st.success(f"💰 Predicted Price: ${price:,.2f}")

        st.subheader("📊 Price Breakdown")
        fig, ax = plt.subplots()
        ax.bar(["Predicted Price"], [price], color="skyblue")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)

    else:
        category_index = int(prediction[0])
        category_label = le.inverse_transform([category_index])[0] if le else str(category_index)
        st.success(f"🏷️ Predicted Category: {category_label}")

        st.subheader("📊 Category Distribution")
        fig, ax = plt.subplots()
        ax.pie([1], labels=[category_label], colors=["lightgreen"], autopct="%1.1f%%")
        st.pyplot(fig)



















