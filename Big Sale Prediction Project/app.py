import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Big Sales Prediction Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Load pipeline safely using binary read mode
@st.cache_resource
def load_pipeline():
    with open('big_sale_pipeline.joblib', 'rb') as f:
        return joblib.load(f)

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model pipeline: {e}")
    st.info("Please run 'python train_big_sale.py' first to train the model and save the pipeline.")

# CSS Styling for Premium UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1A365D;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2B6CB0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛒 Big Sales Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict item outlet sales based on product and store characteristics</div>", unsafe_allow_html=True)

# Sidebar metadata
st.sidebar.image("https://img.icons8.com/clouds/100/000000/shop.png", width=100)
st.sidebar.header("About the Project")
st.sidebar.info(
    "This web application predicts retail sales using an optimized **Gradient Boosting Regressor** pipeline. "
    "Features are scaled and encoded automatically via a Scikit-Learn `ColumnTransformer`."
)
st.sidebar.markdown("**Author:** Nishant Sharma")
st.sidebar.markdown("**Course:** B.Tech (AI & Data Science)")

if model_loaded:
    # Form columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Product Attributes")
        
        item_weight = st.slider("Item Weight (kg)", 4.0, 22.0, 12.8)
        
        item_fat_content = st.selectbox(
            "Item Fat Content",
            options=["Low Fat", "Regular"]
        )
        
        item_visibility = st.slider("Item Visibility", 0.0, 0.35, 0.06)
        
        item_type = st.selectbox(
            "Item Product Category",
            options=[
                "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods", 
                "Dairy", "Baking Goods", "Canned", "Health and Hygiene", "Meat", 
                "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods", 
                "Breakfast", "Seafood"
            ]
        )
        
        item_mrp = st.slider("Item Maximum Retail Price (MRP)", 30.0, 270.0, 140.0)

    with col2:
        st.subheader("🏪 Outlet (Store) Attributes")
        
        outlet_identifier = st.selectbox(
            "Outlet Code",
            options=["OUT027", "OUT013", "OUT049", "OUT046", "OUT035", "OUT045", "OUT018", "OUT017", "OUT019", "OUT010"]
        )
        
        outlet_establishment_year = st.selectbox(
            "Outlet Establishment Year",
            options=[1985, 1987, 1997, 1998, 1999, 2002, 2004, 2007, 2009]
        )
        
        outlet_size = st.selectbox(
            "Outlet Size",
            options=["Small", "Medium", "High"]
        )
        
        outlet_location_type = st.selectbox(
            "Outlet Location Tier",
            options=["Tier 1", "Tier 2", "Tier 3"]
        )
        
        outlet_type = st.selectbox(
            "Outlet Type",
            options=["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
        )

    st.markdown("---")
    
    if st.button("Calculate Predicted Outlet Sales", type="primary", use_container_width=True):
        # Create input dictionary
        input_dict = {
            'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content],
            'Item_Visibility': [item_visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Identifier': [outlet_identifier],
            'Outlet_Establishment_Year': [outlet_establishment_year],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location_type],
            'Outlet_Type': [outlet_type]
        }
        
        # Convert to dataframe
        input_df = pd.DataFrame(input_dict)
        
        # Inference (Log transformations handled automatically inside pipeline by TransformedTargetRegressor)
        prediction = pipeline.predict(input_df)[0]
        
        # Highlight predictions
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-card'>"
            f"  <p style='color: #4A5568; margin-bottom: 0.5rem; font-size: 1.1rem;'>Estimated Outlet Sales</p>"
            f"  <div class='metric-value'>₹ {prediction:,.2f}</div>"
            f"  <p style='color: #718096; font-size: 0.85rem; margin-top: 0.5rem;'>Calculated using optimized Gradient Boosting Regressor</p>"
            f"</div>",
            unsafe_allow_html=True
        )
