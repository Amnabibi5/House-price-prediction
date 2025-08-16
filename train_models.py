# Add this to your training script to save feature alignment artifacts

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

def save_feature_alignment_artifacts(X_encoded, categorical_columns, original_df, target_col):
    """
    Save artifacts needed for exact feature alignment during prediction
    """
    try:
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # 1. Save the exact feature columns used during training
        feature_columns = X_encoded.columns.tolist()
        joblib.dump(feature_columns, 'artifacts/feature_columns.pkl')
        print(f"✅ Saved {len(feature_columns)} feature columns")
        
        # 2. Save categorical mappings for exact reproduction
        categorical_mappings = {}
        for col in categorical_columns:
            if col in original_df.columns and col != target_col:
                unique_values = original_df[col].unique().tolist()
                categorical_mappings[col] = unique_values
                print(f"✅ Saved mapping for {col}: {len(unique_values)} categories")
        
        joblib.dump(categorical_mappings, 'artifacts/categorical_mappings.pkl')
        
        # 3. Save feature information for debugging
        feature_info = {
            'total_features': len(feature_columns),
            'numerical_features': len([col for col in feature_columns if '_' not in col or not any(cat + '_' in col for cat in categorical_columns)]),
            'categorical_features': len([col for col in feature_columns if any(cat + '_' in col for cat in categorical_columns)]),
            'categorical_columns': list(categorical_columns),
            'feature_columns': feature_columns
        }
        joblib.dump(feature_info, 'artifacts/feature_info.pkl')
        
        print("✅ All feature alignment artifacts saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving feature alignment artifacts: {e}")
        return False

def prepare_features_with_alignment_save(df, target_col, is_training=True):
    """
    Prepare features and save alignment artifacts during training
    """
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"📊 Found {len(categorical_columns)} categorical and {len(numerical_columns)} numerical columns")
    
    # Apply one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"🔧 After encoding: {X_encoded.shape[1]} features")
    
    # Save alignment artifacts during training
    if is_training:
        success = save_feature_alignment_artifacts(X_encoded, categorical_columns, df, target_col)
        if not success:
            print("⚠️ Warning: Feature alignment artifacts not saved. Predictions may fail.")
    
    return X_encoded, y, categorical_columns

# Example usage in your main training function:
def train_models_with_alignment():
    """
    Modified training function that saves feature alignment artifacts
    """
    # Load your dataset
    df = pd.read_csv('data/housing.csv')  # Replace with your dataset path
    target_col = 'price'  # Replace with your target column
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target column: {target_col}")
    
    # Prepare features with alignment artifact saving
    X, y, categorical_columns = prepare_features_with_alignment_save(df, target_col, is_training=True)
    
    # Determine task type
    if df[target_col].dtype in ['object', 'category'] or len(df[target_col].unique()) < 10:
        task_type = 'classification'
        # Encode target for classification
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        # Save label encoder
        joblib.dump(label_encoder, 'artifacts/label_encoder.pkl')
        print("✅ Saved label encoder for classification")
    else:
        task_type = 'regression'
        y_encoded = y.values
        print("📈 Regression task detected")
    
    print(f"🔍 Task type: {task_type}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler with feature count verification
    print(f"🔧 Scaler fitted on {scaler.n_features_in_} features")
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    
    # Define models based on task type
    if task_type == 'classification':
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        metrics_columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        metrics_columns = ['Model', 'RMSE', 'MAE', 'R2']
    
    # Train and evaluate models
    results = []
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            if task_type == 'classification':
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append([name, accuracy, precision, recall, f1])
                print(f"✅ {name} - Accuracy: {accuracy:.4f}")
                
            else:  # regression
                from sklearn.metrics import mean_absolute_error
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append([name, rmse, mae, r2])
                print(f"✅ {name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            # Save model
            model_path = f'models/{name}.pkl'
            joblib.dump(model, model_path)
            print(f"💾 Saved model: {model_path}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    # Save metrics
    metrics_df = pd.DataFrame(results, columns=metrics_columns)
    metrics_df.to_csv('artifacts/metrics.csv', index=False)
    print(f"\n📊 Saved metrics to artifacts/metrics.csv")
    print(metrics_df)
    
    # Feature alignment verification
    print(f"\n🎯 Feature Alignment Verification:")
    print(f"   - Original features: {X.shape[1]}")
    print(f"   - Encoded features: {X.shape[1]} → {X_train.shape[1]}")
    print(f"   - Scaler expects: {scaler.n_features_in_} features")
    print(f"   - Categorical columns: {len(categorical_columns)}")
    
    # Test feature alignment with a sample
    try:
        sample_input = {}
        for col in df.columns:
            if col != target_col:
                sample_input[col] = df[col].iloc[0]
        
        # Recreate encoding
        sample_df = pd.DataFrame([sample_input])
        sample_encoded = pd.get_dummies(sample_df, drop_first=True)
        
        # Align with training features
        aligned_features = pd.DataFrame(0, index=[0], columns=X.columns)
        for col in sample_encoded.columns:
            if col in aligned_features.columns:
                aligned_features[col] = sample_encoded[col].iloc[0]
        
        # Test scaling
        sample_scaled = scaler.transform(aligned_features)
        print(f"✅ Feature alignment test passed: {aligned_features.shape}")
        
    except Exception as e:
        print(f"⚠️ Feature alignment test failed: {e}")
    
    print(f"\n🎉 Training complete! All artifacts saved for dashboard compatibility.")
    return metrics_df

if __name__ == "__main__":
    # Run training with feature alignment artifact saving
    results = train_models_with_alignment()
