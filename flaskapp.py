from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

app = Flask(__name__)

# Load the trained LSTM model and scaler
model = pickle.load(open('lstm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    dates = request.form.getlist('date_input')
    
    # Convert the dates to pandas datetime format
    dates = pd.to_datetime(dates)
    
    # Convert the dates to numeric values (ordinal)
    dates = dates.map(pd.Timestamp.toordinal)
    
    # Scale the input data
    scaled_dates = scaler.transform(np.array(dates).reshape(-1, 1))
    
    # Create input sequence for prediction
    input_sequence = scaled_dates[-60:]  # Use the last 60 days of data
    
    # Reshape the input data for LSTM
    input_sequence = np.reshape(input_sequence, (1, input_sequence.shape[0], 1))
    
    # Make the prediction
    prediction = model.predict(input_sequence)
    
    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(prediction)
    
    return render_template('result.html', predicted_price=predicted_price[0][0])

if __name__ == '__main__':
    app.run(debug=True)
