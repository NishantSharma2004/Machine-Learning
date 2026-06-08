# 🛒 Big Sales Prediction using Machine Learning

This project predicts product outlet sales using machine learning techniques.  
It follows an **end-to-end ML pipeline** including data cleaning, preprocessing, model training, evaluation, and saving the final model.

---

## 📌 Problem Statement
Retail businesses want to predict future sales based on product and outlet information.  
This project builds a regression model to predict **Item_Outlet_Sales** using historical data.

---

## 📂 Dataset
- Source: YBI Foundation (Big Sales Data)
- Type: Structured tabular data
- Target Variable: `Item_Outlet_Sales`

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🔄 Project Workflow
1. Data Loading & Exploration
2. Data Cleaning
   - Handling missing values
   - Fixing inconsistent categorical data
   - Handling invalid zero values
3. Feature Engineering
4. Preprocessing using `Pipeline` & `ColumnTransformer`
5. Model Training
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
6. Model Evaluation (RMSE, R² Score)
7. Cross Validation
8. Feature Importance Analysis
9. Model Saving using `joblib`

---

## 🏆 Best Model
- **Random Forest Regressor**
- Selected based on highest R² score and lowest RMSE

---

## 📊 Evaluation Metrics
- RMSE (Root Mean Squared Error)
- R² Score
- Cross-Validation R² Mean

---

## 💾 Saved Model
The trained model is saved using `joblib`.

---

## 🚀 How to Run the Project
1. Clone the repository
2. Install dependencies
3. Open the notebook
4. Run all cells sequentially

---

## 📌 Author
**Nishant Sharma**  
B.Tech (AI & Data Science)  
Machine Learning & Data Analytics Enthusiast

---

## 📜 License
This project is for educational purposes.
