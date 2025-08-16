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

@st.cache_data
def get_training_columns():
    """Get the exact column structure used during training"""
    try:
        # Load a sample model to get feature names
        model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
        if model_files:
            sample_model = joblib.load(f"models/{model_files[0]}")
            if hasattr(sample_model, 'n_features_in_'):
                return sample_model.n_features_in_
        return None
    except:
        return None

def prepare_input_features(input_data, original_df, target_col, expected_features=None):
    """Prepare input features to match training format exactly"""
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
        
        # Create a reference DataFrame with all original columns except target
        reference_df = original_df.drop(target_col, axis=1).iloc[0:1].copy()
        
        # Update reference with user inputs
        for col, value in input_data.items():
            if col in reference_df.columns:
                reference_df[col] = value
        
        # Apply same preprocessing as training (one-hot encoding)
        encoded_df = pd.get_dummies(reference_df, drop_first=True)
        
        # If we know the expected number of features, ensure we match it
        if expected_features and encoded_df.shape[1] != expected_features:
            st.warning(f"⚠️ Feature count mismatch. Model expects {expected_features}, got {encoded_df.shape[1]}")
            # Try to get all possible columns from the entire dataset
            full_encoded = pd.get_dummies(original_df.drop(target_col, axis=1), drop_first=True)
            
            # Create a row with all possible columns, filled with 0s
            template = pd.DataFrame(0, index=[0], columns=full_encoded.columns)
            
            # Update template with encoded values
            for col in encoded_df.columns:
                if col in template.columns:
                    template[col] = encoded_df[col].values[0]
            
            return template
        
        return encoded_df
        
    except Exception as e:
        st.error(f"Error in feature preparation: {str(e)}")
        return None

# 📊 Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["📊 Model Performance", "🎯 Make Predictions", "📈 Data Insights"]
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
                        unique_values = original_df[col].unique().tolist()
                        input_data[col] = st.selectbox(f"{col}", unique_values)
                    elif original_df[col].dtype in ['int64', 'float64']:
                        # Numerical feature with better UX
                        min_val = float(original_df[col].min())
                        max_val = float(original_df[col].max())
                        mean_val = float(original_df[col].mean())
                        
                        # Determine if this should be integer or float
                        if original_df[col].dtype == 'int64':
                            # Integer feature - use slider with integer steps
                            input_data[col] = st.slider(
                                f"{col}", 
                                min_value=int(min_val), 
                                max_value=int(max_val), 
                                value=int(mean_val),
                                step=1
                            )
                        else:
                            # Float feature - use slider with reasonable steps
                            step_size = round((max_val - min_val) / 20, 2)
                            if step_size == 0:
                                step_size = 0.01
                            
                            input_data[col] = st.slider(
                                f"{col}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val,
                                step=step_size,
                                format="%.2f"
                            )
            
            # Make Prediction Button
            if st.button("🔮 Make Prediction", type="primary"):
                try:
                    # Get expected feature count
                    expected_features = get_training_columns()
                    
                    # Prepare features
                    input_features = prepare_input_features(input_data, original_df, target_col, expected_features)
                    
                    if input_features is not None:
                        # Scale features
                        input_scaled = scaler.transform(input_features)
                        
                        # Make prediction
                        prediction = model.predict(input_scaled)[0]
                        
                        # Handle classification vs regression
                        if task_type == "classification":
                            if artifacts['label_encoder']:
                                prediction_label = artifacts['label_encoder'].inverse_transform([prediction])[0]
                                
                                # Get prediction probabilities if available
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(input_scaled)[0]
                                    classes = artifacts['label_encoder'].classes_
                                    
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <h3>🎯 Prediction Result</h3>
                                        <h2>Class: {prediction_label}</h2>
                                        <p>Confidence: {max(probabilities):.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show probability distribution
                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': probabilities
                                    })
                                    
                                    fig_prob = px.bar(
                                        prob_df, 
                                        x='Class', 
                                        y='Probability',
                                        title="Prediction Probabilities"
                                    )
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
                                <h2>{target_col}: {prediction:,.2f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to prepare input features")
                            
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.error("Please check that all features are filled correctly.")
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
            nbins=30
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        value_counts = original_df[target_col].value_counts()
        fig_dist = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Distribution of {target_col}"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature Correlation (for numerical features)
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        st.subheader("🔗 Feature Correlations")
        corr_matrix = original_df[numerical_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature Statistics
    st.subheader("📋 Feature Statistics")
    st.dataframe(original_df.describe(), use_container_width=True)
    
    # Data Sample
    st.subheader("👀 Data Sample")
    st.dataframe(original_df.head(10), use_container_width=True)

# 🔄 Sidebar Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Instructions")
st.sidebar.markdown("""
1. **📊 Model Performance**: View and compare model metrics
2. **🎯 Make Predictions**: Input features and get predictions
3. **📈 Data Insights**: Explore dataset characteristics
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Info")
if metrics_df is not None:
    st.sidebar.markdown(f"**Models Trained**: {len(available_models)}")
    st.sidebar.markdown(f"**Task Type**: {task_type.title()}")
    st.sidebar.markdown(f"**Dataset Size**: {len(original_df)} samples")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 ML Model Comparison Dashboard | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
