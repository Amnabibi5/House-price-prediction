import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 🎨 Page Configuration
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎯 Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .diagnostic-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 Main Title
st.markdown('<h1 class="main-header">🤖 ML Model Comparison Dashboard</h1>', unsafe_allow_html=True)

# 🔧 Helper Functions
@st.cache_data
def load_metrics():
    """Load model performance metrics"""
    try:
        return pd.read_csv("artifacts/metrics.csv")
    except FileNotFoundError:
        st.error("❌ Metrics file not found. Please run the training script first.")
        return None

@st.cache_data
def load_original_data():
    """Load original dataset for feature reference"""
    try:
        return pd.read_csv("data/housing.csv")
    except FileNotFoundError:
        st.error("❌ Original dataset not found.")
        return None

@st.cache_resource
def load_artifacts():
    """Load scaler and label encoder"""
    artifacts = {}
    try:
        artifacts['scaler'] = joblib.load("artifacts/scaler.pkl")
    except FileNotFoundError:
        st.error("❌ Scaler not found.")
        return None
    
    # Try to load label encoder (only exists for classification)
    try:
        artifacts['label_encoder'] = joblib.load("artifacts/label_encoder.pkl")
        artifacts['task_type'] = 'classification'
    except FileNotFoundError:
        artifacts['label_encoder'] = None
        artifacts['task_type'] = 'regression'
    
    return artifacts

@st.cache_data
def load_training_feature_columns():
    """Load the exact feature columns used during training"""
    try:
        return joblib.load("artifacts/feature_columns.pkl")
    except FileNotFoundError:
        st.warning("⚠️ Feature columns file not found. Using fallback method.")
        return None

@st.cache_data
def load_training_categorical_mappings():
    """Load categorical mappings used during training"""
    try:
        return joblib.load("artifacts/categorical_mappings.pkl")
    except FileNotFoundError:
        st.warning("⚠️ Categorical mappings not found. Using original data.")
        return None

def prepare_input_features_exact_match(input_data, original_df, target_col):
    """
    Prepare input features with exact alignment to training features
    This is the most robust method that ensures exact feature matching
    """
    try:
        # Load saved training columns and categorical mappings
        training_columns = load_training_feature_columns()
        categorical_mappings = load_training_categorical_mappings()
        scaler = artifacts['scaler']
        
        if training_columns is None:
            # Fallback: Recreate from original data
            st.warning("Using fallback method - recreating training columns")
            return prepare_input_features_fallback(input_data, original_df, target_col)
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
        
        # Method 1: Use saved categorical mappings if available
        if categorical_mappings is not None:
            st.success("✅ Using saved categorical mappings")
            
            # Apply the exact same encoding as during training
            encoded_df = pd.DataFrame(index=[0])
            
            for col in original_df.columns:
                if col == target_col:
                    continue
                    
                if col in categorical_mappings:
                    # This was a categorical column during training
                    input_value = input_data.get(col, None)
                    
                    # Get all possible encoded columns for this categorical feature
                    possible_columns = [c for c in training_columns if c.startswith(f"{col}_")]
                    
                    # Initialize all categorical columns to 0
                    for encoded_col in possible_columns:
                        encoded_df[encoded_col] = 0
                    
                    # Set the appropriate column to 1 if the value exists
                    if input_value is not None:
                        encoded_col_name = f"{col}_{input_value}"
                        if encoded_col_name in training_columns:
                            encoded_df[encoded_col_name] = 1
                        # If value doesn't exist in training, leave all as 0 (unknown category)
                    
                else:
                    # This was a numerical column
                    if col in training_columns:
                        encoded_df[col] = input_data.get(col, 0)
            
            # Ensure we have all training columns in the correct order
            final_df = pd.DataFrame(0, index=[0], columns=training_columns)
            for col in encoded_df.columns:
                if col in final_df.columns:
                    final_df[col] = encoded_df[col]
            
            return final_df
        
        # Method 2: Standard one-hot encoding with strict alignment
        else:
            st.info("🔄 Using standard encoding with strict alignment")
            
            # Apply one-hot encoding
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            
            # Create a DataFrame with training columns initialized to 0
            aligned_features = pd.DataFrame(0, index=[0], columns=training_columns)
            
            # Fill in the values for columns that exist in input
            for col in input_encoded.columns:
                if col in aligned_features.columns:
                    aligned_features[col] = input_encoded[col].iloc[0]
            
            return aligned_features
            
    except Exception as e:
        st.error(f"❌ Exact match method failed: {str(e)}")
        return prepare_input_features_fallback(input_data, original_df, target_col)

def prepare_input_features_fallback(input_data, original_df, target_col):
    """
    Fallback method when saved artifacts are not available
    """
    try:
        scaler = artifacts['scaler']
        expected_feature_count = scaler.n_features_in_
        
        st.info("🔄 Using fallback feature preparation method")
        
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get numerical and categorical columns from original data
        numerical_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Recreate the same encoding as training
        # First, get the training data encoding structure
        training_sample = original_df.drop(target_col, axis=1).head(1)
        training_encoded = pd.get_dummies(training_sample, drop_first=True)
        training_columns = training_encoded.columns.tolist()
        
        # Ensure we don't exceed the expected feature count
        if len(training_columns) > expected_feature_count:
            training_columns = training_columns[:expected_feature_count]
        
        # Apply same encoding to input
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Align with training columns
        aligned_features = pd.DataFrame(0, index=[0], columns=training_columns)
        
        for col in input_encoded.columns:
            if col in aligned_features.columns:
                aligned_features[col] = input_encoded[col].iloc[0]
        
        # If we still don't have enough features, pad with zeros
        if aligned_features.shape[1] < expected_feature_count:
            missing_count = expected_feature_count - aligned_features.shape[1]
            padding_cols = [f"missing_feature_{i}" for i in range(missing_count)]
            padding_df = pd.DataFrame(0, index=[0], columns=padding_cols)
            aligned_features = pd.concat([aligned_features, padding_df], axis=1)
        
        # If we have too many features, truncate
        elif aligned_features.shape[1] > expected_feature_count:
            aligned_features = aligned_features.iloc[:, :expected_feature_count]
        
        return aligned_features
        
    except Exception as e:
        st.error(f"❌ Fallback method failed: {str(e)}")
        return None

def get_available_models():
    """Get list of available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    return [f.replace('.pkl', '') for f in os.listdir(models_dir) if f.endswith('.pkl')]

def load_model(model_name):
    """Load a specific model"""
    try:
        return joblib.load(f"models/{model_name}.pkl")
    except FileNotFoundError:
        st.error(f"❌ Model {model_name} not found.")
        return None

def validate_input_features(features_df, expected_count, model_name):
    """Validate that features match model expectations"""
    actual_count = features_df.shape[1]
    
    if actual_count == expected_count:
        st.success(f"✅ Perfect feature match for {model_name}: {actual_count} features")
        return True
    elif actual_count < expected_count:
        st.warning(f"⚠️ Feature count mismatch for {model_name}: Got {actual_count}, expected {expected_count}")
        return False
    else:
        st.error(f"❌ Too many features for {model_name}: Got {actual_count}, expected {expected_count}")
        return False

# 📊 Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["📊 Model Performance", "🎯 Make Predictions", "📈 Data Insights", "🔧 System Diagnostics"]
)

# Load necessary data
metrics_df = load_metrics()
original_df = load_original_data()
artifacts = load_artifacts()
available_models = get_available_models()

if metrics_df is None or original_df is None or artifacts is None:
    st.stop()

task_type = artifacts['task_type']
target_col = "price"  # Should match your training script

# 📊 PAGE 1: Model Performance
if page == "📊 Model Performance":
    st.header("📊 Model Performance Comparison")
    
    if not metrics_df.empty:
        # Display metrics table
        st.subheader("📋 Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualize performance
        if task_type == "regression":
            col1, col2 = st.columns(2)
            
            with col1:
                # RMSE Comparison
                fig_rmse = px.bar(
                    metrics_df, 
                    x="Model", 
                    y="RMSE", 
                    title="RMSE Comparison (Lower is Better)",
                    color="RMSE",
                    color_continuous_scale="Reds_r"
                )
                fig_rmse.update_layout(showlegend=False)
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                # R2 Comparison
                fig_r2 = px.bar(
                    metrics_df, 
                    x="Model", 
                    y="R2", 
                    title="R² Score Comparison (Higher is Better)",
                    color="R2",
                    color_continuous_scale="Greens"
                )
                fig_r2.update_layout(showlegend=False)
                st.plotly_chart(fig_r2, use_container_width=True)
                
        else:  # Classification
            # Accuracy Comparison
            fig_acc = px.bar(
                metrics_df, 
                x="Model", 
                y="Accuracy", 
                title="Accuracy Comparison",
                color="Accuracy",
                color_continuous_scale="Blues"
            )
            fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Best Model Highlight
        if task_type == "regression":
            best_model = metrics_df.loc[metrics_df['R2'].idxmax()]
            metric_name = "R² Score"
            metric_value = f"{best_model['R2']:.4f}"
        else:
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
            metric_name = "Accuracy"
            metric_value = f"{best_model['Accuracy']:.4f}"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆 Best Performing Model</h3>
            <h2>{best_model['Model']}</h2>
            <p><strong>{metric_name}:</strong> {metric_value}</p>
        </div>
        """, unsafe_allow_html=True)

# 🎯 PAGE 2: Make Predictions
elif page == "🎯 Make Predictions":
    st.header("🎯 Make Predictions")
    
    if available_models:
        # Model Selection
        selected_model = st.selectbox("🤖 Select Model", available_models)
        
        # Load selected model
        model = load_model(selected_model)
        scaler = artifacts['scaler']
        
        if model and scaler:
            # Show model info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Model", selected_model)
            with col2:
                st.metric("Expected Features", scaler.n_features_in_)
            with col3:
                training_cols = load_training_feature_columns()
                st.metric("Training Columns", len(training_cols) if training_cols else "Unknown")
            
            # Feature alignment status
            if training_cols:
                st.success("✅ Training feature columns available - exact alignment possible")
            else:
                st.warning("⚠️ Training feature columns not found - using fallback method")
            
            st.subheader("📝 Input Features")
            
            # Create input fields based on original dataset
            feature_cols = [col for col in original_df.columns if col != target_col]
            input_data = {}
            
            # Create columns for better layout
            num_cols = 3
            cols = st.columns(num_cols)
            
            for idx, col in enumerate(feature_cols):
                with cols[idx % num_cols]:
                    if original_df[col].dtype in ['object', 'category']:
                        # Categorical feature
                        unique_values = sorted(original_df[col].unique().tolist())
                        input_data[col] = st.selectbox(f"📋 {col}", unique_values, key=f"select_{col}")
                    elif original_df[col].dtype in ['int64', 'float64']:
                        # Numerical feature
                        min_val = float(original_df[col].min())
                        max_val = float(original_df[col].max())
                        mean_val = float(original_df[col].mean())
                        
                        if original_df[col].dtype == 'int64':
                            input_data[col] = st.slider(
                                f"🔢 {col}", 
                                min_value=int(min_val), 
                                max_value=int(max_val), 
                                value=int(mean_val),
                                step=1,
                                key=f"slider_int_{col}"
                            )
                        else:
                            step_size = round((max_val - min_val) / 100, 3)
                            if step_size == 0:
                                step_size = 0.001
                            
                            input_data[col] = st.slider(
                                f"📊 {col}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val,
                                step=step_size,
                                format="%.3f",
                                key=f"slider_float_{col}"
                            )
            
            # Show feature preview
            with st.expander("👀 Input Data Preview"):
                preview_df = pd.DataFrame([input_data])
                st.dataframe(preview_df, use_container_width=True)
            
            # Make Prediction Button
            if st.button("🔮 Make Prediction", type="primary"):
                with st.spinner("Preparing features and making prediction..."):
                    try:
                        # Prepare features with exact matching
                        input_features = prepare_input_features_exact_match(input_data, original_df, target_col)
                        
                        if input_features is not None:
                            # Validate features
                            expected_features = scaler.n_features_in_
                            is_valid = validate_input_features(input_features, expected_features, selected_model)
                            
                            if is_valid:
                                # Scale features
                                input_scaled = scaler.transform(input_features)
                                
                                # Make prediction
                                prediction = model.predict(input_scaled)[0]
                                
                                # Display results based on task type
                                if task_type == "classification":
                                    if artifacts['label_encoder']:
                                        prediction_label = artifacts['label_encoder'].inverse_transform([prediction])[0]
                                        
                                        # Get prediction probabilities if available
                                        if hasattr(model, 'predict_proba'):
                                            probabilities = model.predict_proba(input_scaled)[0]
                                            classes = artifacts['label_encoder'].classes_
                                            max_prob = max(probabilities)
                                            
                                            st.markdown(f"""
                                            <div class="prediction-box">
                                                <h3>🎯 Prediction Result</h3>
                                                <h2>Class: {prediction_label}</h2>
                                                <p>Confidence: {max_prob:.2%}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Show probability distribution
                                            prob_df = pd.DataFrame({
                                                'Class': classes,
                                                'Probability': probabilities
                                            }).sort_values('Probability', ascending=False)
                                            
                                            fig_prob = px.bar(
                                                prob_df, 
                                                x='Class', 
                                                y='Probability',
                                                title="Class Probabilities",
                                                color='Probability',
                                                color_continuous_scale='Blues'
                                            )
                                            fig_prob.update_layout(showlegend=False)
                                            st.plotly_chart(fig_prob, use_container_width=True)
                                        else:
                                            st.markdown(f"""
                                            <div class="prediction-box">
                                                <h3>🎯 Prediction Result</h3>
                                                <h2>Class: {prediction_label}</h2>
                                            </div>
                                            """, unsafe_allow_html=True)
                                else:  # Regression
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <h3>🎯 Prediction Result</h3>
                                        <h2>{target_col}: ${prediction:,.2f}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Show feature importance (if available)
                                if hasattr(model, 'feature_importances_'):
                                    with st.expander("📊 Feature Importance"):
                                        importances = model.feature_importances_
                                        feature_names = load_training_feature_columns()
                                        if feature_names and len(feature_names) == len(importances):
                                            importance_df = pd.DataFrame({
                                                'Feature': feature_names,
                                                'Importance': importances
                                            }).sort_values('Importance', ascending=False).head(10)
                                            
                                            fig_imp = px.bar(
                                                importance_df,
                                                x='Importance',
                                                y='Feature',
                                                orientation='h',
                                                title="Top 10 Feature Importances"
                                            )
                                            fig_imp.update_layout(height=400)
                                            st.plotly_chart(fig_imp, use_container_width=True)
                            else:
                                st.error("❌ Feature validation failed. Please check the training setup.")
                        else:
                            st.error("❌ Failed to prepare input features for prediction.")
                            
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        with st.expander("🔍 Debug Information"):
                            st.code(f"Error Type: {type(e).__name__}")
                            st.code(f"Error Message: {str(e)}")
                            st.code(f"Input Data: {input_data}")
                            if 'input_features' in locals():
                                st.code(f"Feature Shape: {input_features.shape if input_features is not None else 'None'}")
                            
    else:
        st.warning("⚠️ No trained models found. Please run the training script first.")

# 📈 PAGE 3: Data Insights
elif page == "📈 Data Insights":
    st.header("📈 Data Insights")
    
    # Dataset Overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(original_df))
    with col2:
        st.metric("Features", len(original_df.columns) - 1)
    with col3:
        st.metric("Task Type", task_type.title())
    with col4:
        st.metric("Missing Values", original_df.isnull().sum().sum())
    
    # Data Distribution
    st.subheader("📈 Target Variable Distribution")
    
    if task_type == "regression":
        fig_dist = px.histogram(
            original_df, 
            x=target_col, 
            title=f"Distribution of {target_col}",
            nbins=30,
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        value_counts = original_df[target_col].value_counts()
        fig_dist = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {target_col}"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature Analysis
    st.subheader("🔍 Feature Analysis")
    
    # Numerical features analysis
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Numerical Features ({len(numerical_cols)})**")
        if numerical_cols:
            for col in numerical_cols:
                unique_count = original_df[col].nunique()
                st.write(f"- {col}: {unique_count} unique values")
    
    with col2:
        st.write(f"**Categorical Features ({len(categorical_cols)})**")
        if categorical_cols:
            for col in categorical_cols:
                unique_count = original_df[col].nunique()
                st.write(f"- {col}: {unique_count} categories")
    
    # Feature Correlation (for numerical features only)
    if len(numerical_cols) > 1:
        st.subheader("🔗 Numerical Feature Correlations")
        corr_cols = numerical_cols + [target_col] if target_col in original_df.select_dtypes(include=[np.number]).columns else numerical_cols
        corr_matrix = original_df[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature Statistics
    st.subheader("📋 Feature Statistics")
    st.dataframe(original_df.describe(), use_container_width=True)
    
    # Data Sample
    st.subheader("👀 Data Sample")
    st.dataframe(original_df.head(10), use_container_width=True)

# 🔧 PAGE 4: System Diagnostics
elif page == "🔧 System Diagnostics":
    st.header("🔧 System Diagnostics")
    
    # File Structure Check
    st.subheader("📁 File Structure")
    
    directories = ["data", "models", "artifacts"]
    files_status = {}
    
    for directory in directories:
        if os.path.exists(directory):
            files = os.listdir(directory)
            files_status[directory] = {"status": "✅", "files": files, "count": len(files)}
        else:
            files_status[directory] = {"status": "❌", "files": [], "count": 0}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="diagnostic-box">
            <h4>{files_status['data']['status']} Data Directory</h4>
            <p>Files: {files_status['data']['count']}</p>
            <small>{', '.join(files_status['data']['files'][:3])}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="diagnostic-box">
            <h4>{files_status['models']['status']} Models Directory</h4>
            <p>Files: {files_status['models']['count']}</p>
            <small>{', '.join(files_status['models']['files'][:3])}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="diagnostic-box">
            <h4>{files_status['artifacts']['status']} Artifacts Directory</h4>
            <p>Files: {files_status['artifacts']['count']}</p>
            <small>{', '.join(files_status['artifacts']['files'][:3])}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Engineering Diagnostics
    st.subheader("🔧 Feature Engineering Diagnostics")
    
    # Check for critical files
    critical_files = [
        ("feature_columns.pkl", "Training feature columns"),
        ("categorical_mappings.pkl", "Categorical mappings"),
        ("scaler.pkl", "Feature scaler")
    ]
    
    for filename, description in critical_files:
        filepath = f"artifacts/{filename}"
        if os.path.exists(filepath):
            st.success(f"✅ {description}: Available")
        else:
            st.warning(f"⚠️ {description}: Not found (using fallback methods)")
    
    # Model Information
    st.subheader("🤖 Model Information")
    if available_models:
        for model_name in available_models:
            with st.expander(f"📊 {model_name} Details"):
                model = load_model(model_name)
                if model:
                    st.write(f"**Model Type:** {type(model).__name__}")
                    if hasattr(model, 'n_features_in_'):
                        st.write(f"**Expected Features:** {model.n_features_in_}")
                    if hasattr(model, 'classes_'):
                        st.write(f"**Classes:** {list(model.classes_)}")
                    if hasattr(model, 'feature_importances_'):
                        st.write(f"**Feature Importances:** Available")
                    
                    # Model parameters (top 5)
                    st.write("**Key Parameters:**")
                    params = model.get_params()
                    for key, value in list(params.items())[:5]:
                        st.write(f"- {key}: {value}")
                    if len(params) > 5:
                        st.write(f"... and {len(params) - 5} more parameters")
    
    # Feature Alignment Test
    st.subheader("🎯 Feature Alignment Test")
    
    if st.button("🧪 Test Feature Alignment"):
        try:
            # Create a sample input using the first row of original data
            sample_input = {}
            for col in original_df.columns:
                if col != target_col:
                    sample_input[col] = original_df[col].iloc[0]
            
            st.write("**Sample Input:**", sample_input)
            
            # Test feature preparation
            test
