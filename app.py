from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load models
model_ambala = pickle.load(open('model_ambala.pkl', 'rb'))
model_nagpur = pickle.load(open('model_nagpur.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    location = request.form['location'].lower()
    date_str = request.form['date']

    # Determine the model to use
    if 'ambala' in location:
        model = model_ambala
    elif 'nagpur' in location:
        model = model_nagpur
    else:
        return render_template('index.html', prediction_text='Error: Location not supported.')

    # Convert the input date to datetime and extract features
    date = pd.to_datetime(date_str)
    features = [[date.year, date.month, date.day]]

    # Predict the price
    predicted_price = model.predict(features)[0]

    # Return the prediction
    return render_template('index.html', 
                           prediction_text=f'Predicted Price for Spinach at {location.capitalize()} on {date_str}: Rs.{predicted_price:.2f}/Quintal')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get JSON data
    data = request.get_json(force=True)
    location = data['location'].lower()
    date_str = data['date']

    # Determine the model to use
    if 'ambala' in location:
        model = model_ambala
    elif 'nagpur' in location:
        model = model_nagpur
    else:
        return jsonify({'error': 'Location not supported'})

    # Convert the input date to datetime and extract features
    date = pd.to_datetime(date_str)
    features = [[date.year, date.month, date.day]]

    # Predict the price
    predicted_price = model.predict(features)[0]

    return jsonify({'predicted_price': predicted_price})

if __name__ == "__main__":
    app.run(debug=True)
