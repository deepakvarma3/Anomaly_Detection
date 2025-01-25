# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# Load the dataset (Update with your file path if needed)
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Display first few rows of the dataset to understand the structure
df.head()
# Define the feature columns based on the dataset
features = [' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets',
            ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max',
            ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
            'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min']

# Select features and target
X = df[features]
y = df[' Label'].apply(lambda x: 1 if x == 'DDoS' else 0)  # Convert labels to binary

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# Load the dataset (Update with your file path if needed)
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Display first few rows of the dataset to understand the structure
df.head()

from sklearn.preprocessing import MinMaxScaler

# Define the feature columns based on your dataset
features = [' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
            'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min']

# Select the features (X) and target (y)
X = df[features] # df is now available in this scope
y = df[' Label'].apply(lambda x: 1 if x == 'DDoS' else 0)  # Convert labels to binary

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Define features and labels
features = [' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
            'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min']

# Feature matrix (X) and target vector (y)
X = df[features]
y = df[' Label'].apply(lambda x: 1 if x == 'DDoS' else 0)

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
%whos
from sklearn.svm import SVC

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print("Models trained successfully!")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer # Import SimpleImputer for handling NaNs

# Load dataset
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Define features and labels
features = [' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
            'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min']

# Feature matrix (X) and target vector (y)
X = df[features]
y = df[' Label'].apply(lambda x: 1 if x == 'DDoS' else 0)

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean') # Create an imputer instance
X = imputer.fit_transform(X) # Fit and transform to replace NaNs with the mean

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print("Models trained successfully!")

# Evaluate Random Forest model
rf_pred = rf_model.predict(X_test)
print("Random Forest Model Performance:")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))

# Evaluate SVM model
svm_pred = svm_model.predict(X_test)
print("\nSVM Model Performance:")
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))
import joblib

# Save Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Save SVM model
joblib.dump(svm_model, 'svm_model.pkl')

print("Models saved successfully!")
# Load saved models
rf_model_loaded = joblib.load('random_forest_model.pkl')
svm_model_loaded = joblib.load('svm_model.pkl')

# Test predictions with the loaded models
rf_loaded_pred = rf_model_loaded.predict(X_test)
svm_loaded_pred = svm_model_loaded.predict(X_test)

print("Random Forest Loaded Model Test Predictions:", rf_loaded_pred[:10])
print("SVM Loaded Model Test Predictions:", svm_loaded_pred[:10])
import matplotlib.pyplot as plt
import numpy as np

# Feature importance from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrix for Random Forest
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Confusion matrix for SVM
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test)
plt.title("Confusion Matrix - SVM")
plt.show()
# Example new data (replace 'df_new' with your actual DataFrame)
df_new = pd.DataFrame({
    ' Source Port': [12345],
    ' Destination Port': [80],
    ' Protocol': [6],
    ' Flow Duration': [123456],
    ' Total Fwd Packets': [10],
    ' Total Backward Packets': [5],
    'Total Length of Fwd Packets': [300],
    ' Total Length of Bwd Packets': [150],
    ' Fwd Packet Length Max': [150],
    ' Fwd Packet Length Min': [150],
    ' Fwd Packet Length Mean': [150],
    ' Fwd Packet Length Std': [0],
    'Bwd Packet Length Max': [75],
    ' Bwd Packet Length Min': [75],
    ' Bwd Packet Length Mean': [75],
    ' Bwd Packet Length Std': [0],
    ' Flow IAT Mean': [200],
    ' Flow IAT Std': [50],
    ' Flow IAT Max': [300],
    ' Flow IAT Min': [100],
    'Fwd IAT Total': [1000],
    ' Fwd IAT Mean': [100],
    ' Fwd IAT Std': [20],
    ' Fwd IAT Max': [150],
    ' Fwd IAT Min': [50]
})

# Preprocess new data
X_new = imputer.transform(df_new)  # Handle NaN values
X_new_scaled = scaler.transform(X_new)  # Scale features

# Make predictions
rf_prediction = rf_model.predict(X_new_scaled)
svm_prediction = svm_model.predict(X_new_scaled)

print("Random Forest Prediction:", "DDoS" if rf_prediction[0] == 1 else "BENIGN")
print("SVM Prediction:", "DDoS" if svm_prediction[0] == 1 else "BENIGN")
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Define the feature columns
features = [' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
            'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
            ' Fwd IAT Max', ' Fwd IAT Min']

@app.route('/', methods=['POST'])
def predict():
    # Handle file upload
    file = request.files['file']
    df = pd.read_csv(file)

    # Preprocess the data
    imputer = SimpleImputer(strategy='mean')
    X = df[features]
    X = imputer.transform(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict using the models
    rf_pred = rf_model.predict(X_scaled)
    svm_pred = svm_model.predict(X_scaled)

    # Combine predictions
    combined_pred = (rf_pred + svm_pred) >= 1
    combined_pred = ['DDoS' if pred == 1 else 'BENIGN' for pred in combined_pred]

    return jsonify({'predictions': combined_pred})

if __name__ == '__main__':
    app.run(debug=True)
    from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from flask_ngrok import run_with_ngrok # import run_with_ngrok here

app = Flask(__name__) # Define the Flask app here
run_with_ngrok(app)  # Start ngrok when the app starts

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# ... (Rest of your Flask app code)

if __name__ == '__main__':
    app.run() # Run the Flask app
