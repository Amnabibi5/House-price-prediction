import streamlit as st
import pickle

# Title
st.title("🏡 House Price Prediction")
st.write("Enter house details and choose a model to predict the price.")

# Inputs
area = st.number_input("Area (sqft)", min_value=100, max_value=10000, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10)

# Model selection
model_choice = st.selectbox("Choose a model", ["Linear Regression", "KNN", "Random Forest", "SVM"])

# Load scaler
with open('model/scaler.pkl', 'rb') as f:

    scaler = pickle.load(f)

# Load model
model_path = {
    "Linear Regression": "model/linear_regression_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "SVM": "model/svm_model.pkl"
}


try:
    with open(model_path[model_choice], 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file for {model_choice} not found.")
    st.stop()

# Load metrics
with open('model/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Predict
if st.button("Predict Price"):
    input_data = scaler.transform([[area, bedrooms, bathrooms]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"{model_choice} Prediction: ${predicted_price:,.2f}")

    # Display metrics
    st.subheader("📊 Model Performance")
    st.write(f"**R² Score:** {metrics[model_choice]['R²']}")
    st.write(f"**Mean Squared Error:** {metrics[model_choice]['MSE']:,.2f}")
    st.write(f"**F1 Score:** {metrics[model_choice]['F1']}")


