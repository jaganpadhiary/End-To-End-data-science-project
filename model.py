# ------------------------------
# Step 1: Import all libraries
# ------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

# ------------------------------
# Step 2: Load the dataset
# ------------------------------
data = pd.read_csv("housing.csv")

# Display first few rows
print("Sample data:\n", data.head())

# ------------------------------
# Step 3: Check for missing values
# ------------------------------
print("\nMissing values in each column:\n", data.isnull().sum())

# Drop rows with missing values (simple fix for beginners)
data = data.dropna()

# ------------------------------
# Step 4: Convert categorical (text) data into numeric
# ------------------------------
# Identify the column with text data (usually 'ocean_proximity')
if 'ocean_proximity' in data.columns:
    le = LabelEncoder()
    data['ocean_proximity'] = le.fit_transform(data['ocean_proximity'])
    print("\nUnique values in 'ocean_proximity' converted to numbers:", le.classes_)

# ------------------------------
# Step 5: Define input (X) and output (y)
# ------------------------------
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# ------------------------------
# Step 6: Split into train and test sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 7: Train a Linear Regression model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 8: Evaluate the model
# ------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation Results:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# ------------------------------
# Step 9: Save the model
# ------------------------------
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model saved successfully as 'model.pkl'")
