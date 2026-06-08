import streamlit as st
import joblib
import numpy as np

# Set page config for a premium look
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model and mappings
with open('dt_obj.lp', 'rb') as f:
    model = joblib.load(f)

with open('laptop_mappings.joblib', 'rb') as f:
    mappings = joblib.load(f)

# Custom CSS for premium styling
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    p.subtitle {
        text-align: center;
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }
    .stSelectbox label {
        color: #c9d1d9 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(137, 87, 229, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(137, 87, 229, 0.4);
        background: linear-gradient(135deg, #58a6ff 0%, #8957e5 100%);
    }
    .result-container {
        margin-top: 2rem;
        background: rgba(137, 87, 229, 0.08);
        border: 1px dashed #8957e5;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-header {
        font-size: 1rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    .result-price {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(to right, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.write("<h1>Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.write("<p class='subtitle'>Select specifications to estimate the fair retail market value of the laptop.</p>", unsafe_allow_html=True)

# Grid layout for selectboxes
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options=list(mappings['brand'].keys()))
    processor_brand = st.selectbox("Processor Brand", options=list(mappings['processor_brand'].keys()))
    processor_name = st.selectbox("Processor Model", options=list(mappings['processor_name'].keys()))
    processor_gnrtn = st.selectbox("Processor Generation", options=list(mappings['processor_gnrtn'].keys()))
    ram_gb = st.selectbox("RAM Size", options=list(mappings['ram_gb'].keys()))
    ram_type = st.selectbox("RAM Type", options=list(mappings['ram_type'].keys()))
    ssd = st.selectbox("SSD Capacity", options=list(mappings['ssd'].keys()))
    hdd = st.selectbox("HDD Capacity", options=list(mappings['hdd'].keys()))

with col2:
    os_name = st.selectbox("Operating System", options=list(mappings['os'].keys()))
    os_bit = st.selectbox("OS Architecture", options=list(mappings['os_bit'].keys()))
    graphic_card_gb = st.selectbox("Graphic Card (GB)", options=list(mappings['graphic_card_gb'].keys()))
    weight = st.selectbox("Weight / Usage Category", options=list(mappings['weight'].keys()))
    warranty = st.selectbox("Warranty Period", options=list(mappings['warranty'].keys()))
    touchscreen = st.selectbox("Touchscreen Support", options=list(mappings['Touchscreen'].keys()))
    msoffice = st.selectbox("Microsoft Office Pre-installed", options=list(mappings['msoffice'].keys()))

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Price"):
    try:
        # Convert selected labels to integer values using mapping dictionaries
        input_data = [
            mappings['brand'][brand],
            mappings['processor_brand'][processor_brand],
            mappings['processor_name'][processor_name],
            mappings['processor_gnrtn'][processor_gnrtn],
            mappings['ram_gb'][ram_gb],
            mappings['ram_type'][ram_type],
            mappings['ssd'][ssd],
            mappings['hdd'][hdd],
            mappings['os'][os_name],
            mappings['os_bit'][os_bit],
            mappings['graphic_card_gb'][graphic_card_gb],
            mappings['weight'][weight],
            mappings['warranty'][warranty],
            mappings['Touchscreen'][touchscreen],
            mappings['msoffice'][msoffice]
        ]
        
        # Reshape input to 2D array for model prediction
        features = np.array([input_data])
        
        # Run prediction on log scale and convert back to original scale using expm1
        pred_price = float(np.expm1(model.predict(features)[0]))
        
        # Display the result
        st.markdown(f"""
            <div class="result-container">
                <div class="result-header">Estimated Retail Price</div>
                <div class="result-price">₹{pred_price:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
