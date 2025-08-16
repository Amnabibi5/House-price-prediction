# 🏡 House Price Prediction Dashboard

An interactive, production-ready Streamlit app for predicting house prices using machine learning. Built with reproducibility, modularity, and user experience in mind, this project showcases automated model training, feature importance visualization, and real-time predictions—all deployed seamlessly via Streamlit Cloud.

🔗 **Live App**: [Streamlit Dashboard](https://house-price-prediction-tuvtktrzvhkkmc3vh6ztna.streamlit.app/)  
📁 **Source Code**: [GitHub Repository](https://github.com/Amnabibi5/House-price-prediction)

---

## 📌 Project Overview

This dashboard allows users to:
- Input property features and receive instant price predictions
- Compare multiple regression models (e.g., Linear Regression, Random Forest, XGBoost)
- Visualize feature importance and model performance
- Explore a clean, responsive UI with tooltips and expanders for accessibility

Designed for deployment and scalability, the app integrates:
- Modular ML pipelines
- Automated preprocessing and scaling
- Robust error handling and UI polish

---

## 🧠 Machine Learning Workflow

### 🔍 Data Preprocessing
- Feature selection and alignment
- Handling missing values
- Scaling with `StandardScaler`
- Encoding categorical variables (if applicable)

### ⚙️ Model Training
- Multiple models trained and evaluated:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Metrics logged: MAE, RMSE, R²

### 📊 Model Comparison
- Visual comparison of model performance
- Feature importance via SHAP or model coefficients
- Selection of best-performing model for deployment

---

## 🚀 Deployment

- **Platform**: [Streamlit Cloud](https://streamlit.io/cloud)
- **Automation**: GitHub Actions for CI/CD (optional)
- **UI Features**:
  - Expanders for model details
  - Tooltips for feature descriptions
  - Error handling for invalid inputs

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                              |
|------------------|--------------------------------------|
| Python           | Core programming language            |
| Pandas, NumPy    | Data manipulation                    |
| Scikit-learn     | ML models and preprocessing          |
| XGBoost          | Gradient boosting model              |
| Streamlit        | Dashboard development and deployment |
| Matplotlib       | Visualizations                       |

---

## 📂 Repository Structure

├── app.py # Streamlit app entry point ├── models/ # Saved models and scalers ├── utils/ # Helper functions for preprocessing and prediction ├── data/ # Dataset (if included) ├── requirements.txt # Python dependencies ├── README.md # Project documentation

## 📸 Dashboard Preview

![House Price Prediction Dashboard](assets/screenshot.png)

## 🚀 Live Demo

👉 [Streamlit App]((https://house-price-prediction-tuvtktrzvhkkmc3vh6ztna.streamlit.app/)

[![View Demo](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge&logo=streamlit)](https://house-price-prediction-tuvtktrzvhkkmc3vh6ztna.streamlit.app/)





