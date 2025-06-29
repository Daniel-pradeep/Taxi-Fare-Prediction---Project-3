# 🚖 Taxi Fare Prediction App

A Streamlit-based web application that predicts taxi fares using machine learning. Users can select pickup and drop locations on an interactive map, input trip details, and get real-time fare predictions.

## ✨ Features

- **Interactive Map Selection**: Click on the map to select pickup and drop locations
- **Real-time Fare Prediction**: ML model predicts fare based on trip parameters
- **Comprehensive Trip Analysis**: Shows distance, time analysis, and fare breakdown
- **User-friendly Interface**: Clean and intuitive Streamlit interface

## 🛠️ Installation

### Option 1: Quick Setup (Recommended)
```bash
python setup.py
```

### Option 2: Manual Installation
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run main.py
```

## 📋 Requirements

- Python 3.7+
- Required packages (see `requirements.txt`):
  - streamlit
  - streamlit-folium
  - folium
  - pandas
  - numpy
  - scikit-learn
  - pytz
  - xgboost
  - joblib

## 🎯 How to Use

1. **Select Locations**: 
   - Click on the map to select pickup location (green marker)
   - Click again to select drop location (red marker)

2. **Fill Trip Details**:
   - Choose vendor (Uber/OLA)
   - Enter number of passengers
   - Select pickup date and time

3. **Get Prediction**:
   - Click "Submit Booking & Predict Fare"
   - View predicted fare and trip analysis

## 🧠 Model Details

The app uses a trained machine learning model (`modelFinal.pkl`) that predicts taxi fares based on:

### Input Features:
- **VendorID**: Uber (1) or OLA (2)
- **Passenger Count**: Number of passengers
- **Trip Distance**: Calculated using Haversine formula (log-transformed)
- **Pickup Hour**: Hour of the day (0-23)
- **Time Features**: 
  - AM/PM encoding
  - Day of week encoding
  - Weekend flag
  - Night trip flag
  - Rush hour flag

### Model Performance:
- **Algorithm**: Gradient Boosting Regressor
- **R² Score**: ~0.75
- **RMSE**: ~2.80
- **MAE**: ~2.04

## 📊 Output Information

The app provides:
- **Predicted Total Fare**: Main prediction in USD
- **Trip Distance**: Calculated distance in kilometers
- **Trip Analysis**: Day, time period, rush hour, night trip indicators
- **Fare Breakdown**: Estimated base fare, distance fare, and time fare

## 🔧 Technical Details

### Feature Engineering:
- **Haversine Distance**: Calculates great-circle distance between coordinates
- **DateTime Features**: Extracts day, hour, weekend, rush hour, and night indicators
- **Log Transformation**: Applied to trip distance for better model performance
- **Categorical Encoding**: Converts categorical variables to numerical format

### Data Preprocessing:
- Handles timezone conversion (UTC to US/Eastern)
- Applies feature scaling and encoding
- Ensures all inputs match the training data format

## 📁 File Structure

```
├── main.py                 # Main Streamlit application
├── modelFinal.pkl         # Trained ML model
├── requirements.txt       # Python dependencies
├── setup.py              # Installation script
├── README.md             # This file
├── taxi_fare.csv         # Training dataset
└── Taxi_Fare_Regression.ipynb  # Model training notebook
```

## 🚀 Running the App

After installation, the app will be available at `http://localhost:8501`

## 📝 Notes

- The model was trained on NYC taxi data and may need retraining for other locations
- Predictions are estimates and actual fares may vary
- The app assumes US/Eastern timezone for feature extraction
- Make sure `modelFinal.pkl` is in the same directory as `main.py`

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational purposes. 
