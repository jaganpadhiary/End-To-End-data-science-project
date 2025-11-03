# End-To-End-data-science-project
*COMPANY* : CODETECH IT SOLUTIONS

*NAME* : JAGAN PADHIARY

*INTERN ID* : CT04DR268

*DOMAIN* : DATA SCIENCE

*DURATION* : 4 WEEKS
#  Housing Price Prediction Web App

This is an **end-to-end Machine Learning and Flask web application** that predicts **house prices** based on various features such as location, income, and housing details.  
The project uses a **Linear Regression** model trained on the California Housing dataset (`housing.csv`).

---

##  Project Structure

 Housing-Price-Prediction
├── app.py # Flask web application for user input and prediction
├── model.py # Machine Learning model training script
├── model.pkl # Trained Linear Regression model
├── housing.csv # Dataset used for training the model
├── templates/
│ └── index.html # Frontend UI for input and result display
└── README.md # Project documentation


---

##  Project Overview

###  Model Training (`model.py`)
- Loads and cleans the dataset (`housing.csv`)
- Handles missing values
- Encodes categorical data (`ocean_proximity`)
- Trains a **Linear Regression** model
- Evaluates model using **MAE**, **MSE**, and **R² score**
- Saves the trained model as `model.pkl`

###  Web Application (`app.py`)
- Flask-based web app for real-time predictions
- Accepts user inputs through a form
- Loads the saved `model.pkl`
- Predicts and displays the estimated house price instantly

---

##  Features

 --Clean and simple UI using Flask  
 --End-to-end machine learning pipeline  
 --Real-time prediction on web interface  
 -- Beginner-friendly and well-commented Python code  

---

## Technologies Used

| Category | Tools/Frameworks |
|-----------|------------------|
| Programming | Python |
| Web Framework | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Storage | Pickle |
| Visualization (Optional) | Matplotlib |
| Frontend | HTML, CSS (via templates) |

---


Predicted House Price: $245,670.25

## Future Improvements

Add data visualization dashboard

Use advanced ML models (e.g., Random Forest, XGBoost)

Deploy on Render, Heroku, or AWS

Improve frontend using TailwindCSS or Bootstrap


<img width="393" height="423" alt="Image" src="https://github.com/user-attachments/assets/1afa98e4-33d7-4731-8427-c24bcb8fb844" />
