# Big Sales Prediction Project - CRT Folder

This directory contains the optimized preprocessing pipeline, model training script, and interactive Streamlit web application for predicting product outlet sales.

---

## 📂 File Directory

*   `train_big_sale.py`: Pipeline training script that cleans the dataset, sets up imputation/one-hot encoding, and trains a Gradient Boosting Regressor model.
*   `app.py`: Streamlit web dashboard for real-time outlet sales predictions.
*   `big_sale_pipeline.joblib`: Serialized Scikit-Learn pipeline (generated after running the training script).

---

## ⚙️ Technologies Used
*   Python
*   Pandas, NumPy
*   Scikit-Learn (ColumnTransformer, Pipeline, TransformedTargetRegressor, GradientBoostingRegressor)
*   Streamlit (Dashboard)
*   Joblib (Serialization)

---

## 🚀 How to Run

1.  **Run the training script to generate the pipeline:**
    ```bash
    python train_big_sale.py
    ```
    This will load the dataset from the YBI Foundation repository, train the Gradient Boosting model, evaluate its performance, and save the full pipeline as `big_sale_pipeline.joblib`.

2.  **Start the Streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```
    Open your browser and navigate to the address shown in the terminal (typically `http://localhost:8501`).
