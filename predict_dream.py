import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Load trained model, scaler, and dataset
model = joblib.load("dream_cluster_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("clustered_eeg_dreams.csv")

# Define EEG feature columns
X_columns = ['Delta_Mean', 'Theta_Mean', 'Alpha_Mean', 'Beta_Mean', 'Gamma_Mean']

# Get user input
user_values = []
for feature in X_columns:
    while True:
        try:
            value = float(input(f"Enter {feature.replace('_', ' ').lower()}: "))
            user_values.append(value)
            break
        except ValueError:
            print("\u274C Invalid input. Please enter a numeric value.")

# Convert to NumPy array
user_data = np.array(user_values).reshape(1, -1)

# Scale user input
user_data_scaled = scaler.transform(pd.DataFrame(user_data, columns=X_columns))

# Predict cluster
predicted_cluster = model.predict(user_data_scaled)[0]

# Filter dataset for the predicted cluster
cluster_data = df[df['Cluster'] == predicted_cluster]

# If no matching cluster found, find the closest one
if cluster_data.empty:
    print("\u26A0 No exact cluster match found. Searching for the closest available cluster...")
    
    # Find closest cluster using Euclidean distance
    closest_cluster = min(
        df['Cluster'].unique(),
        key=lambda c: euclidean(user_data[0], df[df['Cluster'] == c][X_columns].mean().values)
    )
    
    cluster_data = df[df['Cluster'] == closest_cluster]
    print(f"\U0001F504 Using closest available cluster: {closest_cluster}")


closest_match = min(
    cluster_data.itertuples(),
    key=lambda row: euclidean(user_data[0], [row.Delta_Mean, row.Theta_Mean, row.Alpha_Mean, row.Beta_Mean, row.Gamma_Mean])
)

predicted_dream = closest_match.Dream_Description


print("\n\U0001F4AD Predicted Dream:")
print(predicted_dream)


with open("predicted_dream.txt", "w") as f:
    f.write(predicted_dream)

print("\n✅ Predicted dream saved to 'predicted_dream.txt'")
