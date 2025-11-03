# ------------------------------
# Step 1: Import necessary libraries
# ------------------------------
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# ------------------------------
# Step 2: Load the trained model
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------------------
# Step 3: Create Flask app
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Step 4: Home route (shows form)
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

# ------------------------------
# Step 5: Prediction route (handles form submission)
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    data = {
        'longitude': float(request.form['longitude']),
        'latitude': float(request.form['latitude']),
        'housing_median_age': float(request.form['housing_median_age']),
        'total_rooms': float(request.form['total_rooms']),
        'total_bedrooms': float(request.form['total_bedrooms']),
        'population': float(request.form['population']),
        'households': float(request.form['households']),
        'median_income': float(request.form['median_income']),
        'ocean_proximity': float(request.form['ocean_proximity'])
    }

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Make prediction using trained model
    prediction = model.predict(df)[0]

    # Show result on same page
    result_text = f"🏡 Predicted House Price: ${prediction:,.2f}"

    return render_template('index.html', prediction_text=result_text)

# ------------------------------
# Step 6: Run the app
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
