from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = os.path.join('saved_model', 'xgb_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    print(model.get_booster().feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create input data as a dictionary with the same features used in training
        input_data = {
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude']),
            'wind_mph': float(data['wind_mph']),
            'pressure_mb': float(data['pressure_mb']),
            'humidity': float(data['humidity']),
            'cloud': float(data['cloud']),
            'visibility_km': float(data['visibility_km']),
            'gust_mph': float(data['gust_mph']),
            'condition_Blizzard': 0,
            'condition_Blowing snow': 0,
            'condition_Clear': 0,
            'condition_Cloudy': 1 if data['condition'] == 'Cloudy' else 0,
            'condition_Fog': 0,
            'condition_Freezing drizzle': 0,
            'condition_Freezing fog': 0,
            'condition_Heavy rain': 0,
            'condition_Heavy rain at times': 0,
            'condition_Heavy snow': 0,
            'condition_Light drizzle': 0,
            'condition_Light freezing rain': 0,
            'condition_Light rain': 0,
            'condition_Light rain shower': 0,
            'condition_Light sleet': 0,
            'condition_Light sleet showers': 0,
            'condition_Light snow': 0,
            'condition_Light snow showers': 0,
            'condition_Mist': 0,
            'condition_Moderate or heavy rain in area with thunder': 0,
            'condition_Moderate or heavy rain shower': 0,
            'condition_Moderate or heavy rain with thunder': 0,
            'condition_Moderate or heavy sleet': 0,
            'condition_Moderate or heavy snow in area with thunder': 0,
            'condition_Moderate or heavy snow showers': 0,
            'condition_Moderate rain': 0,
            'condition_Moderate rain at times': 0,
            'condition_Moderate snow': 0,
            'condition_Overcast': 1 if data['condition'] == 'Overcast' else 0,
            'condition_Partly Cloudy': 0,
            'condition_Partly cloudy': 1 if data['condition'] == 'Partly cloudy' else 0,
            'condition_Patchy heavy snow': 0,
            'condition_Patchy light drizzle': 0,
            'condition_Patchy light rain': 0,
            'condition_Patchy light rain in area with thunder': 0,
            'condition_Patchy light rain with thunder': 0,
            'condition_Patchy light snow': 0,
            'condition_Patchy light snow in area with thunder': 0,
            'condition_Patchy moderate snow': 0,
            'condition_Patchy rain nearby': 0,
            'condition_Patchy rain possible': 0,
            'condition_Patchy snow nearby': 0,
            'condition_Patchy snow possible': 0,
            'condition_Sunny': 1 if data['condition'] == 'Sunny' else 0,
            'condition_Thundery outbreaks in nearby': 0,
            'condition_Thundery outbreaks possible': 0,
            'condition_Torrential rain shower': 0
        }

        # Get feature names from the model
        feature_names = model.get_booster().feature_names
        
        # Create input array in the correct order
        input_array = np.array([[input_data[feature] for feature in feature_names]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 1)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
