# 🌾 SoilSense: Smart Crop Recommendation System

SoilSense is a machine learning classification application that recommends the optimal crop to plant based on soil nutrient levels and environmental climate parameters. It features a deployed **Flask web interface** with glassmorphic green gradients and micro-animations.

---

## 📌 Problem Statement
Farmers face difficulties deciding which crop is best suited for their agricultural land due to changing soil nutrients and weather conditions. Planting inappropriate crops leads to low yield. This project trains a classifier to predict one of 22 different crop categories based on nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

---

## 📂 Dataset Details
*   **Source:** Agricultural sensor data.
*   **Features:**
    *   `N`: Nitrogen content in soil (mg/kg).
    *   `P`: Phosphorus content in soil (mg/kg).
    *   `K`: Potassium content in soil (mg/kg).
    *   `temperature`: Climate temperature in Celsius.
    *   `humidity`: Relative humidity in percentage.
    *   `ph`: Soil pH level (0 - 14 scale).
    *   `rainfall`: Average rainfall in mm.
*   **Target Variable:** `label` (Categorical crop name, e.g., rice, maize, mango, grapes, apple).

---

## 🏆 Model Performance
*   **Classifier:** `RandomForestClassifier` (100 Decision Trees).
*   **Performance:** Achieved **99.32% Testing Accuracy** (100% Training Accuracy).
*   **Rationale:** Soil and weather features form complex, non-linear boundaries. Random Forest excels by partitioning data along thresholds and averaging outputs to avoid overfitting.

---

## ⚙️ Technologies Used
*   **Language:** Python
*   **Data Analysis:** Pandas, NumPy
*   **Visualizations:** Seaborn, Matplotlib
*   **Model Training:** Scikit-Learn (Random Forest)
*   **Model Serialization:** Joblib
*   **Web Framework:** Flask
*   **Frontend Styling:** CSS3 (HSL color spaces, glassmorphism card layouts)

---

## 🔄 Project Workflow
1.  **Exploratory Data Analysis (EDA):** Assessed crop distributions and verified soil nutrient bounds.
2.  **Model Training:** Split dataset (80% train, 20% test). Fitted the Random Forest model and computed accuracy scores.
3.  **Model Serialization:** Serialized the classifier to `crop_model.joblib` and label-lookup keys to `crop_mapping.joblib`.
4.  **Flask Microservice:** Created routing inputs. Form values are parsed, validated, fed into the model, and the recommended crop is displayed on the UI.

---

## 🚀 How to Run the Project
1.  Navigate to the project directory:
    ```bash
    cd Crop_Recommendation
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
