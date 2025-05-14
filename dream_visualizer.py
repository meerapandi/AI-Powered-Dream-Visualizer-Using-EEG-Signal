import requests
import os
import webbrowser

# File path for predicted dream
file_path = "predicted_dream.txt"

# Check if the predicted dream file exists
if not os.path.exists(file_path):
    print("❌ No predicted dream found! Please run 'predict_dream.py' first.")
    exit()

# Read the dream description
with open(file_path, "r") as f:
    dream_description = f.read().strip()

print("\n🌙 Predicted Dream Description:")
print(dream_description)
print("\n🖼 Generating image for the dream...")

# DeepAI API key
api_key = "YOUR_API_KEY"# REPLACE WITH YOUR OWN DEEPAI API KEY

# Make the API request
response = requests.post(
    "https://api.deepai.org/api/text2img",
    data={"text": dream_description},
    headers={"api-key": api_key},
    timeout=15
)

# Process the response
if response.status_code == 200:
    data = response.json()
    image_url = data.get("output_url")

    if image_url and image_url.lower() != "none":
        print("\n🖼 Dream Image Generated:", image_url)
        webbrowser.open(image_url)
    else:
        print("⚠ No valid image URL returned. API response:", data)
else:
    print(f"❌ Failed to generate image. Status Code: {response.status_code}")
    print("Response:", response.text)
