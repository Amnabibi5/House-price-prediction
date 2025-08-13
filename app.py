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

    # Sidebar filters
    st.sidebar.subheader("🔍 Filter Data")
    min_price = int(df["price"].min())
    max_price = int(df["price"].max())
    price_range = st.sidebar.slider("Price Range", min_price, max_price, (min_price, max_price))
    bedroom_filter = st.sidebar.multiselect("Bedrooms", sorted(df["bedrooms"].unique()), default=sorted(df["bedrooms"].unique()))

    # Apply filters
    filtered_df = df[
        (df["price"] >= price_range[0]) &
        (df["price"] <= price_range[1]) &
        (df["bedrooms"].isin(bedroom_filter))
    ]

    # Summary statistics
    st.subheader("📌 Summary Statistics")
    st.dataframe(filtered_df.describe())

    # Price distribution
    st.subheader("💰 Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df["price"], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    # Area vs Price
    st.subheader("📐 Area vs Price")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=filtered_df, x="area", y="price", ax=ax2)
    st.pyplot(fig2)

    # Bedrooms vs Price
    st.subheader("🛏 Bedrooms vs Price")
    avg_price_by_bedroom = filtered_df.groupby("bedrooms")["price"].mean().reset_index()
    st.bar_chart(avg_price_by_bedroom.set_index("bedrooms"))

    # Correlation heatmap (numeric only)
    st.subheader("📊 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=["float64", "int64"])
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Model performance
    st.subheader("📈 Model Comparison")
    for model_name, scores in metrics.items():
        st.markdown(f"**{model_name}**")
        st.write(f"R²: {scores['R²']}")
        st.write(f"MSE: {scores['MSE']:,.2f}")
        st.write(f"F1 Score: {scores['F1']}")

    # Feature importance (Random Forest)
    st.subheader("🌲 Feature Importance (Random Forest)")
    rf_model_path = model_path["Random Forest"]
    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)

    features = ["area", "bedrooms", "bathrooms"]
    importances = rf_model.feature_importances_

    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax_imp)
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)












