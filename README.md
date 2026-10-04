# California House Price Prediction Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains an end-to-end Machine Learning Regression project that predicts median house values in California districts[cite: 5]. The model is implemented using Scikit-Learn, utilizing a Random Forest Regressor[cite: 5] and standard feature scaling.

---

## Dataset Notice
*Note: The dataset used in this project is directly fetched via Scikit-Learn's built-in dataset module (`fetch_california_housing`)[cite: 5]. If you need a standalone CSV version for external use, similar California housing datasets can be easily downloaded from Kaggle.*

---

## Dataset Features
The model operates on 8 core demographic and geographic attributes:
* **MedInc**: Median income in block group.
* **HouseAge**: Median house age in block group.
* **AveRooms**: Average number of rooms per household.
* **AveBedrms**: Average number of bedrooms per household.
* **Population**: Block group population.
* **AveOccup**: Average house occupancy.
* **Latitude**: Block group latitude.
* **Longitude**: Block group longitude.
* **MedHouseVal**: Target variable (Median house value)[cite: 5].

---

## Project Workflow
1. **Data Acquisition**: Loading the dataset through `fetch_california_housing`[cite: 5].
2. **Preprocessing**: Feature scaling using `StandardScaler`[cite: 5].
3. **Model Training**: Fitting a `RandomForestRegressor` with fixed random seed for reproducibility[cite: 5].
4. **Evaluation**: Assessing performance via Root Mean Squared Error (RMSE) and $R^2$ Score (~0.80)[cite: 5].
5. **Model Persistence**: Saving the trained estimator and scaler using `joblib`[cite: 5].
6. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/california-house-price-prediction.git](https://github.com/YOUR_USERNAME/california-house-price-prediction.git)
   cd california-house-price-prediction
