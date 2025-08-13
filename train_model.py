import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score
import pickle
import os

# Load dataset
df = pd.read_csv('data/housing.csv')

# Preprocessing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = df[df['price'] < df['price'].quantile(0.95)]
df['price_per_sqft'] = df['price'] / df['area']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['area', 'bedrooms', 'bathrooms']])
X = pd.DataFrame(scaled_features, columns=['area', 'bedrooms', 'bathrooms'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'KNN': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVM': SVR()
}

metrics = {}

# Train and save each model
os.makedirs('model', exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save model
    with open(f'model/{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    y_test_class = (y_test > y.median()).astype(int)
    y_pred_class = (y_pred > y.median()).astype(int)
    f1 = f1_score(y_test_class, y_pred_class)

    metrics[name] = {
        'R²': round(r2, 2),
        'MSE': round(mse, 2),
        'F1': round(f1, 2)
    }

# Save scaler and metrics
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model/metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("✅ Models, scaler, and metrics saved successfully.")
# Display metrics
print("Model Performance Metrics:")
for name, metric in metrics.items():
    print(f"{name}: R²={metric['R²']}, MSE={metric['MSE']}, F1={metric['F1']}")
