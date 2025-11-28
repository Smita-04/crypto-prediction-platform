# 📈 CryptoPredict - AI-Powered Crypto Prediction Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Django](https://img.shields.io/badge/Django-4.0%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**CryptoSight** is a machine learning-based web application that predicts future prices of major cryptocurrencies (Bitcoin, Ethereum, Solana, etc.). Built with **Django** and **TensorFlow**, it utilizes LSTM (Long Short-Term Memory) neural networks to analyze historical data and forecast hourly and daily price trends.

---

## 📸 Project Screenshots

### 1. User Login & Authentication
<img width="1920" height="1080" alt="Screenshot (349)" src="https://github.com/user-attachments/assets/a13ebe21-c06a-4e0e-86c3-b378d4f7e747" />
<img width="1920" height="1080" alt="Screenshot (350)" src="https://github.com/user-attachments/assets/63e24d33-d4ca-4362-9223-6b355424e13f" />
<img width="1920" height="1080" alt="Screenshot (351)" src="https://github.com/user-attachments/assets/ce2d433e-6440-4622-a574-14e475958679" />


### 2. Prediction Dashboard
<!-- Make sure you have a folder named 'screenshots' with an image named 'dashboard.png' -->
<img width="1920" height="1080" alt="Screenshot (285)" src="https://github.com/user-attachments/assets/e1d7742b-40fb-4fe6-a8bc-77fa58c24cb5" />





### 3. Prediction Results & Graphs
![Results Graph](screenshots/prediction.png)

---

## 🚀 Key Features

*   **🤖 AI-Driven Predictions:** Uses pre-trained LSTM models (`.h5`) to forecast prices.
*   **📉 Multi-Timeframe Analysis:** Supports both **Hourly** and **Daily** price predictions.
*   **🪙 Multi-Coin Support:** Covers major assets like BTC, ETH, SOL, ADA, DOT, and LINK.
*   **🔐 User Authentication:** Secure Login and Signup system using Django Auth.
*   **📊 Interactive Visualizations:** Dynamic charts comparing actual historical data vs. predicted values.
*   **📂 Modular Architecture:** Organized into Data Collection, Model Training, and Web Deployment milestones.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python, Django Framework |
| **ML Engine** | TensorFlow, Keras, Scikit-Learn, Joblib |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap |
| **Database** | SQLite (Development) |
| **Version Control** | Git & GitHub |

---

## 📂 Project Structure

The repository is organized by development phases:

```text
CRYPTO-PREDICTION-PLATFORM/
│
├── Milestone1/          # Data Collection Scripts
│   ├── data_collector.py
│   └── ...
│
├── Milestone2/          # Machine Learning Core
│   ├── trainer.py       # LSTM Model Training Logic
│   ├── predictor.py     # Inference Logic
│   └── models/          # Saved .h5 Models & Scalers
│
├── Milestone3/          # Web Application (Django)
│   ├── crypto_web_app/  # Project Settings
│   ├── prediction_app/  # App Logic (Views, URLs)
│   └── templates/       # HTML Frontend
│
├── data/                # Raw CSV Datasets (Daily/Hourly)
├── screenshots/         # Images for README
├── manage.py            # Django Entry Point
└── requirements.txt     # Python Dependencies
