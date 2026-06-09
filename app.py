import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load assets
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('best_random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Delivery Time Predictor", layout="centered")

st.title("🚚 Food Delivery Time Prediction")
st.markdown("Enter the delivery details below to estimate the delivery time in minutes.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    weather = st.selectbox("Weather", encoders['Weather'].classes_)
    traffic = st.selectbox("Traffic Level", encoders['Traffic_Level'].classes_)

with col2:
    prep_time = st.number_input("Preparation Time (min)", min_value=1, max_value=120, value=15)
    time_of_day = st.selectbox("Time of Day", encoders['Time_of_Day'].classes_)
    vehicle = st.selectbox("Vehicle Type", encoders['Vehicle_Type'].classes_)

experience = st.slider("Courier Experience (years)", 0.0, 20.0, 3.0, 0.5)

if st.button("Predict Delivery Time"):
    # Transform inputs
    input_dict = {
        'Distance_km': [distance],
        'Weather': [encoders['Weather'].transform([weather])[0]],
        'Traffic_Level': [encoders['Traffic_Level'].transform([traffic])[0]],
        'Time_of_Day': [encoders['Time_of_Day'].transform([time_of_day])[0]],
        'Vehicle_Type': [encoders['Vehicle_Type'].transform([vehicle])[0]],
        'Preparation_Time_min': [prep_time],
        'Courier_Experience_yrs': [experience]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Prediction
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Delivery Time: {prediction:.2f} minutes")
    st.balloons()