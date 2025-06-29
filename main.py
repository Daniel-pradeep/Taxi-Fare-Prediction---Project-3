import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pytz
import joblib

st.set_page_config(page_title="Taxi Booking", layout="wide")
st.title("🚖 Taxi Booking Form with Map Selection")


@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelFinal.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Haversine distance calculation
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth's radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Feature engineering function
def create_features(pickup_lat, pickup_lon, drop_lat, drop_lon, pickup_datetime, passenger_count, vendor):
    # Calculate trip distance
    trip_distance = haversine(pickup_lon, pickup_lat, drop_lon, drop_lat)
    
    # Convert to datetime and extract features
    pickup_dt = pd.to_datetime(pickup_datetime)
    pickup_dt_edt = pickup_dt.tz_localize('UTC').tz_convert('US/Eastern')
    
    # Extract datetime features
    pickup_day = pickup_dt_edt.day_name()
    is_weekend = pickup_dt_edt.weekday() >= 5
    pickup_hour = pickup_dt_edt.hour
    am_pm = 'AM' if pickup_hour < 12 else 'PM'
    is_night = 1 if (pickup_hour >= 22 or pickup_hour < 6) else 0
    is_rush_hour = 1 if (7 <= pickup_hour <= 9 or 16 <= pickup_hour <= 19) else 0
    
    # Create feature dictionary
    features = {
        'VendorID': 1 if vendor == 'Uber' else 2,  # Assuming 1=Uber, 2=OLA
        'passenger_count': passenger_count,
        'pickup_hour': pickup_hour,
        'trip_distance_log': np.log1p(trip_distance),
        'am_pm_encoded': 0 if am_pm == 'AM' else 1,
        'pickup_day_encoded': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(pickup_day),
        'is_weekend_encoded': 1 if is_weekend else 0,
        'is_night_encoded': is_night,
        'is_rush_hour_encoded': is_rush_hour
    }
    
    return features, trip_distance

# Initialize session state
if 'pickup' not in st.session_state:
    st.session_state.pickup = None
if 'drop' not in st.session_state:
    st.session_state.drop = None
if 'click_stage' not in st.session_state:
    st.session_state.click_stage = 'pickup'

st.subheader("Step 1: Select Pickup and Drop Locations on Map")

# Create base map
m = folium.Map(location=[12.921958, 80.160725], zoom_start=12)

# Add existing markers
if st.session_state.pickup:
    folium.Marker(
        st.session_state.pickup,
        tooltip="Pickup Location",
        icon=folium.Icon(color="green")
    ).add_to(m)

if st.session_state.drop:
    folium.Marker(
        st.session_state.drop,
        tooltip="Drop Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

# Show map and capture click
map_data = st_folium(m, height=500, width=700)

if map_data and map_data.get("last_clicked"):
    latlon = [map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]]

    if st.session_state.click_stage == 'pickup':
        st.session_state.pickup = latlon
        st.session_state.click_stage = 'drop'
        st.success(f"✅ Pickup Location Selected: {latlon}")
    elif st.session_state.click_stage == 'drop':
        st.session_state.drop = latlon
        st.session_state.click_stage = 'done'
        st.success(f"✅ Drop Location Selected: {latlon}")

# Step 2: Show form only if both locations are selected
if st.session_state.pickup and st.session_state.drop:
    st.subheader("Step 2: Fill Out Booking Details")

    with st.form("booking_form"):
        vendor = st.selectbox("Select Vendor", ["Uber", "OLA"])
        num_passengers = st.number_input("Number of Passengers", min_value=1, max_value=10, value=1)
        
        # Allow users to select any future date and time
        pickup_date = st.date_input(
            "Pickup Date", 
            min_value=datetime.today(),
            help="Select the date for your pickup"
        )
        
        # Remove default value to allow user to select any time
        pickup_time = st.time_input(
            "Pickup Time", 
            help="Select the time for your pickup"
        )

        submitted = st.form_submit_button("Submit Booking & Predict Fare")

        if submitted:
            # Validate pickup time is in the future
            pickup_datetime = datetime.combine(pickup_date, pickup_time)
            current_datetime = datetime.now()
            
            if pickup_datetime <= current_datetime:
                st.error("❌ Please select a future pickup time. The selected time has already passed.")
            else:
                # Load model
                model = load_model()
                
                if model is not None:
                    # Create features
                    features, trip_distance = create_features(
                        st.session_state.pickup[0], st.session_state.pickup[1],
                        st.session_state.drop[0], st.session_state.drop[1],
                        pickup_datetime, num_passengers, vendor
                    )
                    
                    # Convert to DataFrame for prediction
                    features_df = pd.DataFrame([features])
                    
                    # Make prediction
                    predicted_fare = model.predict(features_df)[0]
                    
                    # Display results
                    st.success("🚕 Booking Submitted Successfully!")
                    st.write("---")
                    st.subheader("📊 Trip Details & Fare Prediction")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Booking Information:**")
                        st.write(f"• **Vendor:** {vendor}")
                        st.write(f"• **Passengers:** {num_passengers}")
                        st.write(f"• **Pickup Date & Time:** {pickup_date} at {pickup_time}")
                        st.write(f"• **Pickup Location:** {st.session_state.pickup}")
                        st.write(f"• **Drop Location:** {st.session_state.drop}")
                    
                    with col2:
                        st.write("**Trip Analysis:**")
                        st.write(f"• **Distance:** {trip_distance:.2f} km")
                        st.write(f"• **Day of Week:** {pickup_datetime.strftime('%A')}")
                        st.write(f"• **Time Period:** {'AM' if pickup_datetime.hour < 12 else 'PM'}")
                        st.write(f"• **Rush Hour:** {'Yes' if 7 <= pickup_datetime.hour <= 9 or 16 <= pickup_datetime.hour <= 19 else 'No'}")
                        st.write(f"• **Night Trip:** {'Yes' if pickup_datetime.hour >= 22 or pickup_datetime.hour < 6 else 'No'}")
                    
                    st.write("---")
                    st.subheader("💰 Predicted Fare")
                    st.metric(
                        label="Estimated Total Fare",
                        value=f"${predicted_fare:.2f}",
                        delta=f"Based on {trip_distance:.1f} km trip"
                    )
                    
                    # Additional fare breakdown
                    st.write("**Fare Breakdown (Estimated):**")
                    base_fare = predicted_fare * 0.7
                    distance_fare = predicted_fare * 0.2
                    time_fare = predicted_fare * 0.1
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Base Fare", f"${base_fare:.2f}")
                    with col2:
                        st.metric("Distance Fare", f"${distance_fare:.2f}")
                    with col3:
                        st.metric("Time Fare", f"${time_fare:.2f}")
                    
                else:
                    st.error("❌ Model could not be loaded. Please check if modelFinal.pkl exists.")

# Optional: Reset button
st.sidebar.button("🔄 Reset Selections", on_click=lambda: (
    st.session_state.update({'pickup': None, 'drop': None, 'click_stage': 'pickup'})
))