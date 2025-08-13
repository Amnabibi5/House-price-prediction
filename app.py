import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar navigation
page = st.sidebar.selectbox("📂 Select Page", ["🏡 Prediction", "📊 Dashboard"])

# Load scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load metrics
with open('model/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Model paths
model_path = {
    "Linear Regression": "model/linear_regression_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "SVM": "model/svm_model.pkl"
}

# 🏡 Prediction Page
if page == "🏡 Prediction":
    st.title("🏡 House Price Prediction")
    st.write("Enter house details and choose a model to predict the price.")

    # Inputs
    area = st.number_input("Area (sqft)", min_value=100, max_value=10000, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10)

    # Model selection
    model_choice = st.selectbox("Choose a model", ["Linear Regression", "KNN", "Random Forest", "SVM"])

    # Load model
    try:
        with open(model_path[model_choice], 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file for {model_choice} not found.")
        st.stop()

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

# 📊 Dashboard Page
elif page == "📊 Dashboard":
    st.header("📊 House Price Dashboard")

    # Load dataset
    df = pd.read_csv("data/housing.csv")

    # Summary statistics
    st.subheader("📌 Summary Statistics")
    st.dataframe(df.describe())

    # Price distribution
    st.subheader("💰 Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    # Area vs Price
    st.subheader("📐 Area vs Price")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="area", y="price", ax=ax2)
    st.pyplot(fig2)

    # Bedrooms vs Price
    st.subheader("🛏 Bedrooms vs Price")
    avg_price_by_bedroom = df.groupby("bedrooms")["price"].mean().reset_index()
    st.bar_chart(avg_price_by_bedroom.set_index("bedrooms"))

    # Model performance
    st.subheader("📈 Model Comparison")
    for model_name, scores in metrics.items():
        st.markdown(f"**{model_name}**")
        st.write(f"R²: {scores['R²']}")
        st.write(f"MSE: {scores['MSE']:,.2f}")
        st.write(f"F1 Score: {scores['F1']}")








