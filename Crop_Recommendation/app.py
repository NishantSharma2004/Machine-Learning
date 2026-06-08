from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model and mapping dictionary with binary mode to avoid local encoding bugs
with open('crop_model.joblib', 'rb') as f:
    model = joblib.load(f)

with open('crop_mapping.joblib', 'rb') as f:
    mapping_data = joblib.load(f)

reverse_dict = mapping_data['reverse_dict']

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve soil and environmental metrics from form
            n = float(request.form['N'])
            p = float(request.form['P'])
            k = float(request.form['K'])
            temp = float(request.form['temperature'])
            hum = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rain = float(request.form['rainfall'])

            # Prepare input data for prediction
            features = np.array([[n, p, k, temp, hum, ph, rain]])

            # Predict and map target ID to crop name
            pred_id = int(model.predict(features)[0])
            crop_name = reverse_dict.get(pred_id, "Unknown Crop")
            
            # Capitalize name for cleaner UI
            crop_name = crop_name.capitalize()

            return render_template('predict.html', prediction=crop_name, inputs=request.form)
        except Exception as e:
            return render_template('predict.html', error=f"Invalid Input: {str(e)}")

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
