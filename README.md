# 📈 CryptoPredict - AI-Powered Crypto Prediction Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Django](https://img.shields.io/badge/Django-4.0%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Celery](https://img.shields.io/badge/Celery-Async_Tasks-green)
![Redis](https://img.shields.io/badge/Redis-Message_Broker-red)
![Binance API](https://img.shields.io/badge/API-Binance_RealTime-yellow)

**CryptoPredict**  is an advanced machine learning platform designed to predict cryptocurrency price movements. Unlike static predictors, this application leverages **Celery and Redis** to handle asynchronous tasks and fetches **real-time data via the Binance API** to provide up-to-the-minute forecasts.

The system is trained on extensive historical data (sourced from **CoinDesk**) dating back to the inception of each coin, ensuring robust model performance.

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
<img width="1920" height="1080" alt="Screenshot (286)" src="https://github.com/user-attachments/assets/55a859ab-1690-4b22-a2e2-cddcb9591cd3" />

<img width="1920" height="1080" alt="Screenshot (297)" src="https://github.com/user-attachments/assets/7ecaa6f2-528f-4100-86c0-84d0035dbb23" />

<img width="1920" height="1080" alt="Screenshot (298)" src="https://github.com/user-attachments/assets/327098a8-8ac8-4164-8248-ad70d0153262" />


---

## 🚀 Key Features

*   **🧠 Deep Learning Models:** Utilizes **LSTM (Long Short-Term Memory)** and **GRU** neural networks, optimized for time-series forecasting.
*   **⚡ Real-Time Predictions:** Connects to the **Binance API** to fetch live market data and generate instant price predictions.
*   **📚 Extensive Historical Data:** Models are trained on datasets sourced from **CoinDesk**, covering the entire history of each coin (from launch to present).
*   **🔄 Asynchronous Processing:** Implements **Celery** workers with a **Redis** broker to handle data fetching and model inference in the background without freezing the UI.
*   **🪙 9 Major Currencies:** Fully supported predictions for:
    *   Bitcoin (BTC)
    *   Ethereum (ETH)
    *   Solana (SOL)
    *   Cardano (ADA)
    *   Polkadot (DOT)
    *   Chainlink (LINK)
    *   Dogecoin (DOGE)
    *   Litecoin (LTC)
    *   Polygon (MATIC)
*   **🐧 WSL Environment:** Developed and optimized using **Windows Subsystem for Linux (WSL)** for superior performance and compatibility with Redis/Celery.

---


## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend Framework** | Django (Python) | Core web server and routing. |
| **ML Engine** | TensorFlow / Keras | Training LSTM models and running inference. |
| **Task Queue** | **Celery** | Handling background jobs (scheduled data fetching). |
| **Message Broker** | **Redis** | Communication between Django and Celery. |
| **Live Data Source** | **Binance API** | Real-time candle/price data. |
| **Historical Data** | **CoinDesk API** | Training datasets (2014-Present). |
| **Environment** | **WSL (Ubuntu)** | Development environment. |
| **Frontend** | HTML5, Bootstrap, JS | Responsive user interface and charts. |

---

## 📂 Detailed Project Structure

## ⚙️ Installation & Setup (WSL/Linux)

**Note:** Since this project utilizes **Redis** (as a message broker) and **Celery** (for asynchronous tasks), it is strictly recommended to run this project inside **WSL (Windows Subsystem for Linux)** or a native Linux environment (Ubuntu/Debian).

### 1. Prerequisites
Ensure you have the following installed in your WSL/Linux environment:
*   **Python 3.8+**
*   **Git**
*   **Redis Server:**
    ```bash
    sudo apt-get install redis-server
    ```

### 2. Clone the Repository
```bash
git clone https://github.com/Smita-04/crypto-prediction-platform.git
cd crypto-prediction-platform
