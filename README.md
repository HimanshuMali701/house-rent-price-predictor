🏠 House Rent Price Predictor

An end-to-end Machine Learning web application built using Streamlit that predicts house rent prices and compares outputs from multiple regression models including Random Forest, Decision Tree, and Support Vector Regressor (SVR).

The application is designed to simulate a real-world ML system, providing not just predictions but also model comparison, insights, confidence range, and visual analysis.

🚀 Live Demo


https://your-app-name.streamlit.app

📌 Key Features

🔢 Predicts monthly house rent based on property details

🌳 Model comparison across:

Random Forest Regressor

Decision Tree Regressor

Support Vector Regressor (SVR)

📊 Visual comparison of model predictions

📈 Confidence range for predicted rent

🧠 Model insights & explanations (interview-friendly)

🖥️ Clean, responsive Streamlit UI

☁️ Deployable on Streamlit Cloud

🧠 Machine Learning Models Used
Model	Description	R² Score
Random Forest	Ensemble-based, robust, best generalization	0.9549
Decision Tree	Interpretable, fast but high variance	0.9015
SVR	Sensitive to scaling, smooth predictions	0.8800

Evaluation Metrics:

R² Score

Mean Squared Error (MSE)

📊 Features Used for Prediction

BHK (Bedrooms, Hall, Kitchen)

Size (Sq. Ft.)

Number of Bathrooms

City

Furnishing Status


Numerical features are scaled using training data statistics, and categorical features are one-hot encoded.

🛠️ Tech Stack

Programming Language: Python

Libraries:

Pandas

NumPy

Streamlit

ML Models:

Random Forest Regressor

Decision Tree Regressor

Support Vector Regressor

Deployment: Streamlit Cloud

📁 Project Structure
house-rent-price-predictor/
│
├── app.py              
├── House_Rent_prediction.ipynb
├── House_Rent_Dataset.csv
├── README.md           

🎯 Use Cases

🏘️ Helps tenants estimate fair rent prices

🏢 Assists property owners in pricing decisions

📊 Demonstrates ML model comparison in practice

🎓 Ideal portfolio project for Data Science / ML internships

📈 Future Enhancements

✅ Integrate real trained models using joblib

✅ Add SHAP explainability plots

✅ City-wise rent heatmaps

✅ Downloadable PDF prediction report

✅ User authentication & history tracking