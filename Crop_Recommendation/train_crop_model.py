import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset from GitHub...")
url = "https://raw.githubusercontent.com/upflairs-pvt-ltd/3rd_july_datascience/master/farmer_guider/farmer.csv"
data = pd.read_csv(url)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Encode label mapping
labal_dict = {
    'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
    'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
    'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
    'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
    'jute': 20, 'coffee': 21
}
data['label'] = data['label'].map(labal_dict)

# Split features and target
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Model Training Accuracy: {train_acc * 100:.2f}%")
print(f"Model Testing Accuracy: {test_acc * 100:.2f}%")

# Save model and dictionary mappings
print("Saving model to crop_model.joblib...")
joblib.dump(model, 'crop_model.joblib')

# Save mapping dictionaries
mapping_data = {
    'label_dict': labal_dict,
    'reverse_dict': {v: k for k, v in labal_dict.items()}
}
joblib.dump(mapping_data, 'crop_mapping.joblib')
print("Model and mapping saved successfully!")
