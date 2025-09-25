This project is a Streamlit web application that predicts food quality using sensor data (temperature, moisture, airflow, vibration).
The model is trained using Random Forest Classifier and stored as food_quality_model.pkl.

⚡ Features
Simulates or streams sensor data
Predicts food quality (good/bad) using trained ML model
Stores records in MongoDB Atlas
Role-based login (Operator / Manager)
Simple UI built with Streamlit

Example Workflow
Sensor data is generated or streamed
Model (food_quality_model.pkl) predicts probability of good/bad food
Predictions + sensor data are stored in MongoDB Atlas
Managers can view quality statistics & reports

Tech Stack
Python 3.13
Streamlit (UI)
Scikit-learn (ML model)
Pandas, NumPy (Data processing)
MongoDB Atlas (Database)
