import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("clustered_eeg_dreams.csv")

# Extract features and labels
X = df[['Delta_Mean', 'Theta_Mean', 'Alpha_Mean', 'Beta_Mean', 'Gamma_Mean']]
y = df['Cluster']

# Encode cluster labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Label encoder saved successfully!")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")  # Save the scaler
print("✅ Scaler saved successfully!")

# Train XGBoost Classifier
model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump(model, "dream_cluster_model.pkl")  # Save the model
print("✅ Model trained and saved successfully!")

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"🔥 Model Accuracy: {accuracy*100:.2f}%")
