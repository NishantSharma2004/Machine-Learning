import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model():
    print("Loading dataset from YBI Repository...")
    # Load dataset
    url = "https://github.com/YBI-Foundation/Dataset/raw/main/Big%20Sales%20Data.csv"
    df = pd.read_csv(url)
    
    # 1. Clean categorical variations in Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })
    
    # 2. Impute missing Item_Weight values using mean grouped by Item_Type
    df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
    
    # 3. Features and Target selection
    X = df.drop(['Item_Identifier', 'Item_Outlet_Sales'], axis=1)
    y = df['Item_Outlet_Sales']
    
    # Define features
    numeric_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
    categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    
    # 4. Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 5. Model Training (Optimized Gradient Boosting Regressor on Raw Target)
    gbr = GradientBoostingRegressor(n_estimators=30, learning_rate=0.15, max_depth=6, random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', gbr)
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2529)
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance Evaluation:")
    print(f"R2 Score: {r2 * 100:.2f}%")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # 7. Save the pipeline
    joblib.dump(pipeline, 'big_sale_pipeline.joblib')
    print("\nPipeline successfully saved as 'big_sale_pipeline.joblib'.")

if __name__ == "__main__":
    train_model()
