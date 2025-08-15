import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load label encoder for classification models
try:
    le = joblib.load("label_encoder.pkl")
except:
    le = None

# Model options
model_names = [
    "LinearRegression", "Ridge", "Lasso", "RandomForestRegressor", "GradientBoostingRegressor",
    "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "SVC"
]

# Detect model type
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
model = joblib.load(f"{selected_model}.pkl")

# Input form
st.title("🏠 House Price Prediction Dashboard")
st.write("Enter house specifications below:")

with st.form("prediction_form"):
    # Replace with your actual features
    area = st.number_input("Area (sq ft)", min_value=100)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    submitted = st.form_submit_button("Predict")

# Prepare input
input_df = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "location": [location]
})

# Prediction
if submitted:
    prediction = model.predict(input_df)

    if is_regression(selected_model):
        price = prediction[0]
        st.success(f"💰 Predicted Price: ${price:,.2f}")

        # Visualization: bar chart
        st.subheader("📊 Price Breakdown")
        fig, ax = plt.subplots()
        ax.bar(["Predicted Price"], [price], color="skyblue")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)

    else:
        category_index = int(prediction[0])
        category_label = le.inverse_transform([category_index])[0] if le else str(category_index)
        st.success(f"🏷️ Predicted Category: {category_label}")

        # Visualization: pie chart
        st.subheader("📊 Category Distribution")
        fig, ax = plt.subplots()
        ax.pie([1], labels=[category_label], colors=["lightgreen"], autopct="%1.1f%%")
        st.pyplot(fig)














