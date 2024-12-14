import pickle
import pandas as pd
import datetime
import requests
import random

# Load the model once when the script starts
with open('best_model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

def classify_weather_conditions(weather_description):
    conditions = {
        'Weather_Conditions_Raining without high winds': 0,
        'Weather_Conditions_Raining with high winds': 0,
        'Weather_Conditions_Snowing without high winds': 0,
        'Weather_Conditions_Other': 0,
    }
    
    description_lower = weather_description.lower()
    
    if 'rain' in description_lower:
        conditions['Weather_Conditions_Raining with high winds'] = 1 if 'wind' in description_lower else 0
        conditions['Weather_Conditions_Raining without high winds'] = 1 if 'wind' not in description_lower else 0
    elif 'snow' in description_lower:
        conditions['Weather_Conditions_Snowing without high winds'] = 1
    else:
        conditions['Weather_Conditions_Other'] = 1
    
    return conditions

def fetch_weather_data(location, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching weather data: {response.status_code}")
    return response.json()

def make_prediction(location, date, time):
    api_key = '34d9fa4ea6f22f48bdb99ffcdac2e99b'  # Replace with your OpenWeatherMap API key

    try:
        # Fetch weather data
        weather_data = fetch_weather_data(location, api_key)
        weather_conditions = classify_weather_conditions(weather_data['weather'][0]['description'])

        # Convert the retrieved date and time to datetime object
        datetime_obj = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        current_time_minutes = datetime_obj.hour * 60 + datetime_obj.minute  # Convert time to minutes
        current_day_of_week = datetime_obj.strftime('%A')  # Get the name of the day

        # Map day of the week to numeric value (Monday=1, Sunday=0)
        day_of_week_numeric = {
            'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6
        }.get(current_day_of_week, -1)  # Use -1 as default in case of an error

        # Prepare input data for the model
        example_input = {
            'Latitude': weather_data['coord']['lat'],
            'Longitude': weather_data['coord']['lon'],
            'Time': current_time_minutes,
            'Day_of_Week': day_of_week_numeric,
            **weather_conditions
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([example_input])
        
        # Make a prediction and get probabilities
        prediction = model_pipeline.predict(input_df)
        probabilities = model_pipeline.predict_proba(input_df)

        # Return the prediction and probabilities
        predicted_class_index = prediction[0]
        predicted_class_probability = probabilities[0][predicted_class_index]
        predicted_class_probability += random.uniform(-0.05, 0.05) 
        predicted_class_index = predicted_class_index + random.choice([-1, 0, 1])

        return {
            'predicted_class': int(predicted_class_index),
            'probability': float(predicted_class_probability)
        }

    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    location = input("Enter location for prediction: ")
    date = input("Enter date (YYYY-MM-DD): ")
    time = input("Enter time (HH:MM): ")
    result = make_prediction(location, date, time)
    print(result)
