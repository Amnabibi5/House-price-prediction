import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# Load data
df = pd.read_csv("Housing.csv")

# Create price category for classification
df["price_category"] = pd.cut(df["price"], bins=[0, 5000000, 10000000, np.inf],
                              labels=["Low", "Medium", "High"])

# Targets
y_reg = df["price"]
y_clf = df["price_category"]

# Features
X = df.drop(columns=["price", "price_category"])
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Regression models
regression_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor()
}

# Classification models
classification_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "SVC": SVC()
}

# Save performance metrics
results = []

# Train regression models
for name, model in regression_models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"{name}.pkl")

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({"model": name, "type": "regression", "score": rmse})

# Encode classification target
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)
joblib.dump(le, "label_encoder.pkl")

# Train classification models
for name, model in classification_models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf_encoded, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"{name}.pkl")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"model": name, "type": "classification", "score": acc})

# Save metrics
pd.DataFrame(results).to_csv("metrics.csv", index=False)


# Create synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Train model
model = LinearRegression()
model.fit(X, y)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save model as .pkl
with open("models/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/linear_model.pkl")


