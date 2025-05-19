AI Powered Dream Visualizer Using EEG Signal

An innovative AI-powered system that visualizes dreams based on EEG brainwave signals using machine learning and image generation APIs.

## 🚀 Project Overview

This project predicts and visualizes dreams using EEG signal data. It uses the mean values of five EEG brainwaves (Alpha, Beta, Gamma, Delta, Theta) to classify the type of dream a user might be experiencing. Then, it generates a visual representation of that dream using the DeepAI image generation API.

## 📺 Demo

[![Watch the video](https://img.youtube.com/vi/ZvQsrTi5ijg/hqdefault.jpg)

## 🧩 Features

- Input EEG wave means: alpha, beta, gamma, delta, theta.
- Predict dream description using a trained XGBoost model (accuracy: **96.3%**).
- Generate dream visuals using the predicted description via DeepAI's image generator.
- Save predictions and automatically open the visual in a web browser.

## 🗂️ Project Structure

```bash
dream_project/
│
├── clustered_eeg_dreams.csv        # Dataset with EEG means and dream labels
├── train_model.py                  # Trains XGBoost model and saves encoders
├── predict_dream.py                # User inputs EEG values → model predicts dream
├── dream_visualizer.py            # Generates image using DeepAI API
│
├── dream_cluster_model.pkl         # Trained XGBoost model
├── scaler.pkl                      # StandardScaler object
├── label_encoder.pkl               # LabelEncoder for dream labels
├── predicted_dream.txt             # Stores latest predicted dream
├── dream_venv/                     # Python virtual environment
