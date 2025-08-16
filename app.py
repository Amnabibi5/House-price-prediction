import streamlit as st
import pandas as pd
import joblib
import os

# Load model and label encoder
model_path = "models/LinearRegression.pkl"
encoder_path = "models/label_encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    st.error("❌ Required model files not found. Please ensure 'LinearRegression.pkl' and 'label_encoder.pkl' exist in the 'models' folder.")
    st.stop()

model = joblib.load(model_path)
location_encoder = joblib.load(encoder_path)

st.title("🏠 House Price Prediction App")

# ℹ️ Help Section
with st.expander("ℹ️ What do these inputs mean?"):
    st.markdown("""
    - **📐 Area**: Total covered area in square feet  
    - **🛏 Bedrooms**: Number of bedrooms  
    - **🛁 Bathrooms**: Number of bathrooms  
    - **🏢 Stories**: Number of floors  
    - **🚗 Parking**: Number of parking spaces  
    - **🛋 Guest Room**: Is there a guest room?  
    - **🏚 Basement**: Is there a basement?  
    - **🔥 Hot Water Heating**: Installed or not  
    - **❄️ Air Conditioning**: Available or not  
    - **🌟 Preferred Area**: Located in a premium zone  
    - **🪑 Furnishing Status**: Furnished, semi, or unfurnished  
    - **🛣 Main Road Access**: Direct access to main road  
    - **📍 Location**: Neighborhood or city area
    """)

# Input form
st.header("Enter House Details")

area = st.number_input("📐 Area (sq ft)", min_value=500, max_value=10000, step=50, help="Total covered area of the house in square feet")
bedrooms = st.selectbox("🛏 Number of Bedrooms", [1, 2, 3, 4, 5], help="Total number of bedrooms in the house")
bathrooms = st.selectbox("🛁 Number of Bathrooms", [1, 2, 3, 4], help="Total number of bathrooms in the house")
stories = st.selectbox("🏢 Number of Stories", [1, 2, 3], help="How many floors the house has")
parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3], help="Number of dedicated parking spots")
guestroom = st.selectbox("🛋 Guest Room", ["Yes", "No"], help="Does the house include a guest room?")
basement = st.selectbox("🏚 Basement", ["Yes", "No"], help="Is there a basement in the house?")
hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["Yes", "No"], help="Is hot water heating installed?")
airconditioning = st.selectbox("❄️ Air Conditioning", ["Yes", "No"], help="Is air conditioning available?")
prefarea = st.selectbox("🌟 Preferred Area", ["Yes", "No"], help="Is the house located in a preferred residential zone?")
furnishingstatus = st.selectbox("🪑 Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"], help="Level of furnishing provided with the house")
mainroad = st.selectbox("🛣 Main Road Access", ["Yes", "No"], help="Is the house directly accessible from a main road?")
location = st.selectbox("📍 Location", location_encoder.classes_, help="Select the neighborhood or city area")

# Predict button
if st.button("Predict Price"):
    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "location": location,
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

    input_df = pd.DataFrame([input_dict])

    # Map binary features
    binary_map = {"Yes": 1, "No": 0}
    for col in ["guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "mainroad"]:
        input_df[col] = input_df[col].map(binary_map)

    # Encode location
    input_df["location"] = location_encoder.transform(input_df["location"])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: PKR {prediction:,.0f}")





