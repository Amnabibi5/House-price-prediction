import streamlit as st
import pandas as pd
import joblib
import os

# Define models
regression_models = {
    "Linear Regression": "models/LinearRegression.pkl"
}

classification_models = {
    "KNN": "models/KNN.pkl",
    "Random Forest": "models/RandomForestClassifier.pkl",
    "SVM": "models/SVM.pkl"
}

# Load label encoder if needed
def load_encoder(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

location_encoder = load_encoder("models/location_encoder.pkl")

# Input form
def get_user_input():
    st.subheader("📋 House Features")
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
    location = st.selectbox("📍 Location", location_encoder.classes_ if location_encoder else ["Unknown"])

    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
        "mainroad": mainroad,
        "location": location
    }

    return pd.DataFrame([input_dict])

# Preprocess input
def preprocess_input(df):
    binary_map = {"Yes": 1, "No": 0}
    for col in ["guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "mainroad"]:
        df[col] = df[col].map(binary_map)

    if location_encoder:
        df["location"] = location_encoder.transform(df["location"])

    df = df.astype({
        "area": "float64",
        "bedrooms": "int64",
        "bathrooms": "int64",
        "stories": "int64",
        "parking": "int64",
        "guestroom": "int64",
        "basement": "int64",
        "hotwaterheating": "int64",
        "airconditioning": "int64",
        "prefarea": "int64",
        "mainroad": "int64",
        "location": "int64"
    })

    df["furnishingstatus"] = df["furnishingstatus"].astype(str)
    return df

# Predict
def predict_with_models(models, input_df, task="regression"):
    results = {}
    for name, path in models.items():
        if not os.path.exists(path):
            results[name] = "❌ Model file not found"
            continue
        try:
            model = joblib.load(path)
            prediction = model.predict(input_df)[0]
            results[name] = prediction
        except Exception as e:
            results[name] = f"❌ Error: {e}"
    return results

# Main app
st.title("🔍 House Model Comparison App")

tab1, tab2 = st.tabs(["📈 Regression", "🧠 Classification"])

with tab1:
    st.header("Compare Regression Models")
    input_df = get_user_input()
    input_df = preprocess_input(input_df)
    if st.button("🔍 Predict Price"):
        results = predict_with_models(regression_models, input_df)
        for model, output in results.items():
            st.write(f"**{model}**: {output if isinstance(output, str) else f'PKR {output:,.0f}'}")

with tab2:
    st.header("Compare Classification Models")
    input_df = get_user_input()
    input_df = preprocess_input(input_df)
    if st.button("🔍 Predict Category"):
        results = predict_with_models(classification_models, input_df, task="classification")
        for model, output in results.items():
            st.write(f"**{model}**: {output}")



