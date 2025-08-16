from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# 📁 Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# 📥 Load dataset
df = pd.read_csv("from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# 📁 Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# 📥 Load dataset
df = pd.read_csv("data/house_data.csv")


# 🧠 Define your target column
target_col = "price"  # 🔁 Change this to match your dataset

# ✅ Validate target column
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")


# 🧼 Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# 🔄 Split features and target
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# 🧠 Detect task type
task_type = "regression" if y.dtype in [np.float64, np.int64] and len(np.unique(y)) > 10 else "classification"

# 🔠 Encode target if classification
if task_type == "classification":
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, "artifacts/label_encoder.pkl")


# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📏 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "artifacts/scaler.pkl")

# 🤖 Define models
models = {
    "regression": {
        "LinearRegression": LinearRegression()
    },
    "classification": {
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True)
    }
}

# 📈 Train and evaluate
metrics = []

for name, model in models[task_type].items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)


    # 💾 Save model
    joblib.dump(model, f"models/{name}.pkl")


    # 📊 Evaluate
    if task_type == "regression":
        rmse = mean_squared_error(y_test, y_pred) # Removed squared=False
        r2 = r2_score(y_test, y_pred)
        metrics.append({"Model": name, "RMSE": rmse, "R2": r2})
    else:
        acc = accuracy_score(y_test, y_pred)
        metrics.append({"Model": name, "Accuracy": acc})

# 📁 Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("artifacts/metrics.csv", index=False)

print(f"✅ Training complete.\n📦 Models saved in /models\n📊 Metrics saved in /artifacts")")


# 🧠 Define your target column
target_col = "price"  # 🔁 Change this to match your dataset

# ✅ Validate target column
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")


# 🧼 Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# 🔄 Split features and target
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# 🧠 Detect task type
task_type = "regression" if y.dtype in [np.float64, np.int64] and len(np.unique(y)) > 10 else "classification"

# 🔠 Encode target if classification
if task_type == "classification":
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, "artifacts/label_encoder.pkl")


# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📏 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "artifacts/scaler.pkl")

# 🤖 Define models
models = {
    "regression": {
        "LinearRegression": LinearRegression()
    },
    "classification": {
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True)
    }
}

# 📈 Train and evaluate
metrics = []

for name, model in models[task_type].items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)


    # 💾 Save model
    joblib.dump(model, f"models/{name}.pkl")


    # 📊 Evaluate
    if task_type == "regression":
        rmse = mean_squared_error(y_test, y_pred) # Removed squared=False
        r2 = r2_score(y_test, y_pred)
        metrics.append({"Model": name, "RMSE": rmse, "R2": r2})
    else:
        acc = accuracy_score(y_test, y_pred)
        metrics.append({"Model": name, "Accuracy": acc})

# 📁 Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("artifacts/metrics.csv", index=False)

print(f"✅ Training complete.\n📦 Models saved in /models\n📊 Metrics saved in /artifacts")










