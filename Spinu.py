import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the data
data = pd.read_csv('SPINACH.csv')

# Drop unnecessary columns
data = data.drop(columns=['Min Price', 'Max Price', 'Market Name'])

# Ensure the 'Date' column is properly formatted
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Separate data for Nagpur and Ambala
nagpur_df = data[data['District Name'].str.contains('Nagpur', case=False)]
ambala_df = data[data['District Name'].str.contains('Ambala', case=False)]

# Function to process data, train model, and save it
def train_model(df, model_filename):
    # Extract features from the date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Features and target variable
    X = df[['Year', 'Month', 'Day']]
    y = df['Modal Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model using pickle
    pickle.dump(model, open(model_filename, 'wb'))

    print(f"Model saved to {model_filename}")
    return model

# Train models for Ambala and Nagpur
model_ambala = train_model(ambala_df, 'model_ambala.pkl')
model_nagpur = train_model(nagpur_df, 'model_nagpur.pkl')

# Predict the modal price for a given date
def predict_price(date_str, model_filename):
    # Load the model
    model = pickle.load(open(model_filename, 'rb'))

    # Convert the input date to datetime
    date = pd.to_datetime(date_str, format='%d-%m-%Y')

    # Extract features
    features = [[date.year, date.month, date.day]]

    # Predict the price
    predicted_price = model.predict(features)
    return predicted_price[0]

# Example usage
input_date = input("Enter a date (dd-mm-yyyy): ")
predicted_price_ambala = predict_price(input_date, 'model_ambala.pkl')
print(f"Predicted Modal Price for Ambala: {predicted_price_ambala}")

predicted_price_nagpur = predict_price(input_date, 'model_nagpur.pkl')
print(f"Predicted Modal Price for Nagpur: {predicted_price_nagpur}")
