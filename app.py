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
def load_saved_feature_columns():
    """Load saved feature columns from training if available"""
    try:
        return joblib.load("artifacts/feature_columns.pkl")
    except FileNotFoundError:
        return None

@st.cache_data
def get_training_feature_columns(original_df, target_col):
    """Get the exact feature columns used during training by recreating the preprocessing"""
    try:
        # First try to load saved feature columns
        saved_columns = load_saved_feature_columns()
        if saved_columns is not None:
            return saved_columns
            
        # If not available, recreate the exact preprocessing from training
        df_encoded = pd.get_dummies(original_df, drop_first=True)
        feature_columns = df_encoded.drop(target_col, axis=1).columns.tolist()
        return feature_columns
    except Exception as e:
        st.error(f"Error getting training columns: {str(e)}")
        return None

def prepare_input_features_fixed(input_data, original_df, target_col):
    """Prepare input features to match training format exactly"""
    try:
        # Step 1: Get the expected feature columns from training
        expected_columns = get_training_feature_columns(original_df, target_col)
        if expected_columns is None:
            st.error("❌ Could not determine expected feature columns")
            return None
            
        # Step 2: Create a single-row DataFrame with user inputs
        input_row = pd.DataFrame([input_data])
        
        # Step 3: Apply one-hot encoding (same as training)
        input_encoded = pd.get_dummies(input_row, drop_first=True)
        
        # Step 4: Create a DataFrame with all expected columns, filled with 0s
        aligned_features = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        # Step 5: Fill in the values for columns that exist in input_encoded
        for col in input_encoded.columns:
            if col in aligned_features.columns:
                aligned_features[col] = input_encoded[col].iloc[0]
        
        st.success(f"✅ Features aligned: {aligned_features.shape[1]} columns")
        return aligned_features
        
    except Exception as e:
        st.error(f"❌ Error in feature preparation: {str(e)}")
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
            # Feature columns info
            expected_columns = get_training_feature_columns(original_df, target_col)
            
            # Show model info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Selected Model", selected_model)
            with col2:
                st.metric("Expected Features", len(expected_columns) if expected_columns else "Unknown")
            with col3:
                st.metric("Scaler Features", artifacts['scaler'].n_features_in_)
            
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
                        input_data[col] = st.selectbox(f"📋 {col}", unique_values, key=f"select_{col}")
                    elif original_df[col].dtype in ['int64', 'float64']:
                        # Numerical feature with better UX
                        min_val = float(original_df[col].min())
                        max_val = float(original_df[col].max())
                        mean_val = float(original_df[col].mean())
                        
                        # Determine if this should be integer or float
                        if original_df[col].dtype == 'int64':
                            # Integer feature - use slider with integer steps
                            input_data[col] = st.slider(
                                f"🔢 {col}", 
                                min_value=int(min_val), 
                                max_value=int(max_val), 
                                value=int(mean_val),
                                step=1,
                                key=f"slider_int_{col}"
                            )
                        else:
                            # Float feature - use slider with reasonable steps
                            step_size = round((max_val - min_val) / 20, 2)
                            if step_size == 0:
                                step_size = 0.01
                            
                            input_data[col] = st.slider(
                                f"📊 {col}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val,
                                step=step_size,
                                format="%.2f",
                                key=f"slider_float_{col}"
                            )
            
            # Make Prediction Button
            if st.button("🔮 Make Prediction", type="primary"):
                try:
                    # Debug: Show input data
                    with st.expander("🔍 Debug Info"):
                        st.write("**Input Data:**", input_data)
                        st.write("**Input Data Types:**", {k: type(v).__name__ for k, v in input_data.items()})
                    
                    # Prepare features with fixed alignment
                    input_features = prepare_input_features_fixed(input_data, original_df, target_col)
                    
                    if input_features is not None:
                        # Verify feature count matches scaler expectations
                        expected_features = scaler.n_features_in_
                        actual_features = input_features.shape[1]
                        
                        st.info(f"🔍 Feature Check: Expected {expected_features}, Got {actual_features}")
                        
                        if actual_features != expected_features:
                            st.error(f"❌ Feature mismatch: Expected {expected_features}, got {actual_features}")
                            st.info("💡 This might be due to categorical variables not being encoded properly during training vs prediction.")
                            
                            # Show feature comparison
                            with st.expander("🔧 Feature Analysis"):
                                if expected_columns:
                                    st.write("**Expected columns:**", expected_columns[:10], "..." if len(expected_columns) > 10 else "")
                                st.write("**Current columns:**", input_features.columns.tolist()[:10], "..." if len(input_features.columns) > 10 else "")
                                
                        else:
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
                                            title="Prediction Probabilities",
                                            color='Probability',
                                            color_continuous_scale='Blues'
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
                                    <h2>{target_col}: ${prediction:,.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show prediction confidence interval (optional)
                                st.info(f"💡 Predicted {target_col}: ${prediction:,.2f}")
                    else:
                        st.error("❌ Failed to prepare input features")
                            
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.error("Please check that all features are filled correctly.")
                    # Show more detailed error info
                    with st.expander("🔧 Error Details"):
                        st.code(f"Error Type: {type(e).__name__}")
                        st.code(f"Error Message: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
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
    
    # Feature Correlation (for numerical features)
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        st.subheader("🔗 Feature Correlations")
        corr_matrix = original_df[numerical_cols].corr()
        
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
                        st.write(f"**Classes:** {model.classes_}")
                    
                    # Model parameters
                    st.write("**Model Parameters:**")
                    params = model.get_params()
                    for key, value in list(params.items())[:5]:
                        st.write(f"- {key}: {value}")
                    if len(params) > 5:
                        st.write(f"... and {len(params) - 5} more parameters")
    
    # Feature Engineering Info
    st.subheader("🔧 Feature Engineering")
    
    if artifacts:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scaler Information:**")
            st.write(f"- Type: {type(artifacts['scaler']).__name__}")
            st.write(f"- Features: {artifacts['scaler'].n_features_in_}")
            st.write(f"- Feature Names: {'Available' if hasattr(artifacts['scaler'], 'feature_names_in_') else 'Not Available'}")
        
        with col2:
            if artifacts['label_encoder']:
                st.write("**Label Encoder Information:**")
                st.write(f"- Classes: {len(artifacts['label_encoder'].classes_)}")
                st.write(f"- Class Names: {list(artifacts['label_encoder'].classes_)}")
            else:
                st.write("**Label Encoder:** Not used (Regression task)")
    
    # Feature Columns Analysis
    expected_columns = get_training_feature_columns(original_df, target_col)
    if expected_columns:
        st.subheader("📋 Feature Columns Analysis")
        st.write(f"**Total Features After Encoding:** {len(expected_columns)}")
        
        # Categorize columns
        categorical_cols = [col for col in expected_columns if any(orig_col in col for orig_col in original_df.select_dtypes(include=['object', 'category']).columns)]
        numerical_cols = [col for col in expected_columns if col not in categorical_cols]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Numerical Features:** {len(numerical_cols)}")
            if numerical_cols:
                st.write(numerical_cols[:10])
                if len(numerical_cols) > 10:
                    st.write(f"... and {len(numerical_cols) - 10} more")
        
        with col2:
            st.write(f"**Categorical Features:** {len(categorical_cols)}")
            if categorical_cols:
                st.write(categorical_cols[:10])
                if len(categorical_cols) > 10:
                    st.write(f"... and {len(categorical_cols) - 10} more")

# 🔄 Sidebar Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Instructions")
st.sidebar.markdown("""
1. **📊 Model Performance**: View and compare model metrics
2. **🎯 Make Predictions**: Input features and get predictions  
3. **📈 Data Insights**: Explore dataset characteristics
4. **🔧 System Diagnostics**: Check system status and debug issues
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ System Info")
if metrics_df is not None:
    st.sidebar.markdown(f"**Models Trained**: {len(available_models)}")
    st.sidebar.markdown(f"**Task Type**: {task_type.title()}")
    st.sidebar.markdown(f"**Dataset Size**: {len(original_df)} samples")

# System Status
st.sidebar.markdown("### 🚦 System Status")
status_items = [
    ("Data", "✅" if original_df is not None else "❌"),
    ("Models", "✅" if available_models else "❌"),  
    ("Artifacts", "✅" if artifacts else "❌"),
    ("Metrics", "✅" if metrics_df is not None else "❌")
]

for item, status in status_items:
    st.sidebar.markdown(f"{status} {item}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 ML Model Comparison Dashboard | Built with Streamlit | "
    f"Task: {task_type.title()} | Models: {len(available_models)}"
    "</div>", 
    unsafe_allow_html=True
)
