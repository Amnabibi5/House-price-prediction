import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# 📁 Load models and artifacts
model_dir = "models"
artifact_dir = "artifacts"
scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))

# 🧠 Load label encoder (optional)
label_encoder_path = os.path.join(artifact_dir, "label_encoder.pkl")
label_encoder = joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None

# 📊 Load metrics
metrics_path = os.path.join(artifact_dir, "metrics.csv")
metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()

# 📐 Load feature columns and ensure it's a list
feature_cols_path = os.path.join(artifact_dir, "feature_columns.pkl")
if os.path.exists(feature_cols_path):
    raw_cols = joblib.load(feature_cols_path)
    feature_cols = raw_cols.tolist() if isinstance(raw_cols, np.ndarray) else raw_cols
else:
    feature_cols = None

# 📦 Model lists
regression_models = [f for f in os.listdir(model_dir) if "LinearRegression" in f]
classification_models = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and f not in regression_models]

# 🎨 App layout
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction Dashboard")
tab1, tab2 = st.tabs(["📈 Regression Models", "🧠 Classification Models"])

# 📋 Input form
def get_user_input(prefix=""):
    st.subheader("📋 Enter House Details")
    area = st.number_input("📐 Area (sq ft)", min_value=500, max_value=10000, step=50, key=f"{prefix}_area")
    bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5], key=f"{prefix}_bedrooms")
    bathrooms = st.selectbox("🛁 Bathrooms", [1, 2, 3, 4], key=f"{prefix}_bathrooms")
    stories = st.selectbox("🏢 Stories", [1, 2, 3], key=f"{prefix}_stories")
    parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3], key=f"{prefix}_parking")
    guestroom = st.selectbox("🛋 Guest Room", ["Yes", "No"], key=f"{prefix}_guestroom")
    basement = st.selectbox("🏚 Basement", ["Yes", "No"], key=f"{prefix}_basement")
    hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["Yes", "No"], key=f"{prefix}_hotwaterheating")
    airconditioning = st.selectbox("❄️ Air Conditioning", ["Yes", "No"], key=f"{prefix}_airconditioning")
    prefarea = st.selectbox("🌟 Preferred Area", ["Yes", "No"], key=f"{prefix}_prefarea")
    furnishingstatus = st.selectbox("🪑 Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"], key=f"{prefix}_furnishingstatus")
    mainroad = st.selectbox("🛣 Main Road Access", ["Yes", "No"], key=f"{prefix}_mainroad")

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
        "mainroad": mainroad
    }
    return pd.DataFrame([input_dict])

# 🧼 Preprocessing
def preprocess_input(df):
    binary_map = {"Yes": 1, "No": 0}
    for col in ["guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "mainroad"]:
        df[col] = df[col].map(binary_map)

    df["furnishingstatus"] = df["furnishingstatus"].astype(str)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # ✅ FIX: Align with training columns or scaler fallback
    expected_cols = feature_cols if feature_cols else getattr(scaler, "feature_names_in_", None)

    if expected_cols:
        # Add missing columns
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        # Drop unexpected columns
        df_encoded = df_encoded[[col for col in expected_cols if col in df_encoded.columns]]

        # 🧪 Debug info
        st.write("🧪 Input columns:", df_encoded.columns.tolist())
        st.write("🧪 Expected columns:", expected_cols)
        missing = set(expected_cols) - set(df_encoded.columns)
        extra = set(df_encoded.columns) - set(expected_cols)
        if missing:
            st.warning(f"🚨 Missing columns: {missing}")
        if extra:
            st.warning(f"🚨 Unexpected columns: {extra}")
    else:
        st.error("❌ No feature column reference available. Please check your artifacts.")

    # Ensure float dtype
    df_encoded = df_encoded.astype(float)
    df_scaled = scaler.transform(df_encoded)
    return df_scaled, df_encoded

# 📊 Feature Importance Plot
def plot_feature_importance(model, model_name, input_df):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = input_df.columns
    elif hasattr(model, "coef_"):
        importances = model.coef_
        features = input_df.columns
    else:
        st.info(f"ℹ️ Feature importance not available for {model_name}")
        return

    st.subheader(f"🧠 Feature Importance: {model_name}")
    fig, ax = plt.subplots()
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(features)[sorted_idx], np.array(importances)[sorted_idx], color="lightgreen")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

# 🔍 Predict and display
def predict_and_display(models, input_scaled, input_encoded, task):
    predictions = {}
    for model_file in models:
        model_name = model_file.replace(".pkl", "")
        model_path = os.path.join(model_dir, model_file)
        try:
            model = joblib.load(model_path)
            pred = model.predict(input_scaled)[0]
            if task == "classification" and label_encoder:
                pred = label_encoder.inverse_transform([int(pred)])[0]
            predictions[model_name] = pred

            with st.expander(f"📊 Feature Importance for {model_name}"):
                plot_feature_importance(model, model_name, input_encoded)
        except Exception as e:
            predictions[model_name] = f"❌ Error: {e}"

    st.subheader("🔮 Predictions")
    for name, value in predictions.items():
        if task == "classification":
            st.write(f"**{name}**: {value}")
        else:
            st.write(f"**{name}**: PKR {float(value):,.0f}" if isinstance(value, (int, float, np.generic)) else value)

    # ✅ Fix for np.number issue
    numeric_preds = {k: float(v) for k, v in predictions.items() if isinstance(v, (int, float, np.generic))}
    if len(numeric_preds) == len(predictions):
        st.subheader("📊 Model Comparison")
        fig, ax = plt.subplots()
        ax.bar(numeric_preds.keys(), numeric_preds.values(), color="skyblue")
        ax.set_ylabel("Price (PKR)" if task == "regression" else "Class")
        ax.set_title("Predicted Output by Model")
        st.pyplot(fig)

    if not metrics_df.empty:
        st.subheader("📐 Model Performance Metrics")
        filtered = metrics_df[metrics_df["Model"].isin(predictions.keys())]
        st.dataframe(filtered.style.format({"RMSE": "{:,.0f}", "R2": "{:.2f}", "Accuracy": "{:.2f}"}))

# 🧠 Tabs
with tab1:
    input_df = get_user_input(prefix="reg")
    input_scaled, input_encoded = preprocess_input(input_df)
    if st.button("🔍 Predict Price", key="predict_reg"):
        predict_and_display(regression_models, input_scaled, input_encoded, task="regression")

with tab2:
    input_df = get_user_input(prefix="cls")
    input_scaled, input_encoded = preprocess_input(input_df)
    if st.button("🔍 Predict Category", key="predict_cls"):
        predict_and_display(classification_models, input_scaled, input_encoded, task="classification")

