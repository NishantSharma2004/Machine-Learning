# 💻 Laptop Price Estimation using Machine Learning

This project estimates laptop prices based on technical specifications. It features feature extraction from unstructured strings, target variable normalization (log-transform), a **Gradient Boosting Regressor** model, and an interactive **Streamlit web application**.

---

## 📌 Problem Statement
Laptop pricing is determined by CPU and GPU architectures, memory capacity, storage types, screen resolution, and operating systems. Due to high-end models (gaming laptops, Macbooks), price target variables are right-skewed. This project extracts cleaner columns from text and trains an optimized regressor to predict values in INR.

---

## 📂 Dataset Details
*   **Features:**
    *   `Ram`: Memory size in GB (numeric).
    *   `Weight`: Laptop weight in kg (numeric).
    *   `Touchscreen` & `Ips`: Screen attributes extracted from descriptions (binary).
    *   `Cpu_brand`: CPU manufacturer/tier (e.g. Intel Core i5, AMD Ryzen).
    *   `Gpu_brand`: GPU manufacturer (e.g. Nvidia, Intel, AMD).
    *   `SSD` & `HDD`: Storage capacity in GB (numeric).
    *   `OpSys`: Operating system (e.g. Windows, macOS).
*   **Target Variable:** `Price` (continuous numeric value in INR).

---

## 🏆 Model Performance & Log-Transform
*   **Log-Transform:** Budget laptops are concentrated at low price ranges while premium machines are outliers. Training on raw prices yields high regression errors. Applying `np.log1p(Price)` normalizes the target distribution. Model outputs are reverted to normal scales using `np.expm1(y_pred)`.
*   **Algorithm:** `GradientBoostingRegressor` (150 estimators, 0.1 learning rate).
*   **R² Performance:** Boosted testing R² to **81.2%** on log scale.

---

## ⚙️ Technologies Used
*   **Language:** Python
*   **Data Analysis:** Pandas, NumPy
*   **Model Training:** Scikit-Learn (Gradient Boosting Regressor)
*   **Model Serialization:** Joblib
*   **Web Dashboard:** Streamlit (with secure dropdown widgets to prevent input validation errors)

---

## 🔄 Project Workflow
1.  **Feature Extraction:** Cleaned string attributes using Pandas. Parsed the 'Cpu' column into 'Cpu_brand', extracted SSD/HDD capacities, and mapped screen resolutions.
2.  **Log-Scaling:** Normalized price targets using `np.log1p`.
3.  **Model Training:** Split dataset (80/20). Trained the Gradient Boosting model.
4.  **Serialization:** Saved model weights to `model.joblib`.
5.  **Streamlit App:** Configured dropdown widgets mapping to model category mappings. Runs user inputs on backend prediction and displays the calculated price.

---

## 🚀 How to Run the Project
1.  Navigate to the project directory:
    ```bash
    cd Laptop_Price_Prediction
    ```
2.  Install dependencies:
    ```bash
    pip install streamlit joblib scikit-learn pandas numpy
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  Open your browser and navigate to the address shown in the terminal (typically `http://localhost:8501`).

---

## 📌 Author
*   **Nishant Sharma** - B.Tech in Artificial Intelligence & Data Science
