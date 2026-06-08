# 🏍️ Used Bike Price Prediction using Machine Learning

This project predicts the resale price of used bikes based on historical market data. It features a supervised regression pipeline and a deployed **Flask web application** that accepts user parameters and renders live valuations in real time.

---

## 📌 Problem Statement
Resale pricing of motorbikes depends on several factors like wear and tear (mileage), age, engine displacement (power), brand equity, and ownership history. This project trains a regression model to estimate the continuous variable `price` based on these inputs.

---

## 📂 Dataset Details
*   **Size:** 32,000+ records of used bikes sold in India.
*   **Features:**
    *   `kms_driven`: Total distance traveled (numeric).
    *   `age`: Age of the bike in years (numeric).
    *   `power`: Engine displacement in cc (numeric).
    *   `brand`: Bike manufacturer (categorical; mapped to integer codes).
    *   `owner`: Number of previous owners (categorical; mapped to ordinal ranks).
*   **Target Variable:** `price` (continuous numeric value in INR).

---

## ⚙️ Technologies Used
*   **Language:** Python
*   **Data Analysis:** Pandas, NumPy
*   **Model Training:** Scikit-Learn (Random Forest Regressor / Linear Regression)
*   **Model Serialization:** Joblib
*   **Web Framework:** Flask
*   **Web Design:** HTML5, CSS3 (Custom styles)

---

## 🔄 Project Workflow
1.  **Data Cleaning:** Selected key fields, dropped duplicate records to avoid training bias, and filtered outliers.
2.  **Feature Engineering:** Mapped categorical variables (`brand`, `owner`) to dense integer values. Ordinal ownership status (e.g. First Owner -> 1, Second Owner -> 2) preserved logical progression.
3.  **Model Training:** Fitted a `RandomForestRegressor` and evaluated performance using R² score and Root Mean Squared Error (RMSE).
4.  **Serialization:** Saved the model parameters as `model.joblib`.
5.  **Flask Microservice:** Created an API endpoint to receive user requests, decode categories using dictionary mappings, feed parameters to the model, and display the predicted price on the frontend.

---

## 💾 Saved Model Deployment Fix
To prevent standard `joblib.load()` text-mode decode errors in Python 3.13 environments, the serialized binary file is loaded securely via an open binary stream:
```python
with open('model.joblib', 'rb') as f:
    model = joblib.load(f)
```

---

## 🚀 How to Run the Project
1.  Navigate to the project directory:
    ```bash
    cd Used_Bike_Price_Prediction
    ```
2.  Install dependencies:
    ```bash
    pip install flask joblib scikit-learn pandas numpy
    ```
3.  Run the Flask application:
    ```bash
    python app.py
    ```
4.  Open your browser and navigate to:
    ```text
    http://127.0.0.1:5001/
    ```

---

## 📌 Author
*   **Nishant Sharma** - B.Tech in Artificial Intelligence & Data Science
