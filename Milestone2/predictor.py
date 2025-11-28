import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import joblib

import tensorflow as tf
from keras.models import load_model

# --- TA IMPORTS (Must match trainer.py exactly) ---
# These imports are critical for replicating the feature engineering.
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD as MACDIndicator
from ta.volatility import AverageTrueRange as ATRIndicator
from ta.volume import OnBalanceVolumeIndicator
# --- END TA IMPORTS ---

# --- Robust Path Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming predictor.py is in 'Milestone2', project_root is one level up
project_root = os.path.abspath(os.path.join(script_dir, '..'))
OUTPUT_MODELS_PATH = os.path.join(project_root, 'models') # This path points to crypto-prediction-platform/models

# --- GLOBAL TIMESTEPS (Must match trainer.py exactly) ---
COMMON_TIMESTEPS_DAILY = 30
COMMON_TIMESTEPS_HOURLY = 24

# --- Helper function to convert INR to USD (Copied from trainer.py) ---
def convert_inr_to_usd(inr_value):
    """Fetches current INR to USD exchange rate and converts the given INR value."""
    if inr_value is None:
        return None
    try:
        response = requests.get("https://open.er-api.com/v6/latest/INR")
        response.raise_for_status()
        data = response.json()
        usd_rate = data['rates']['USD']
        return inr_value * usd_rate
    except requests.exceptions.RequestException as e:
        print(f"Error fetching INR to USD exchange rate: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing exchange rate data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during INR to USD conversion: {e}")
        return None

# --- Helper function to convert USD to INR (Copied from trainer.py) ---
def convert_usd_to_inr(usd_value):
    """Fetches current USD to INR exchange rate and converts the given USD value."""
    if usd_value is None:
        return None
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD")
        response.raise_for_status()
        data = response.json()
        inr_rate = data['rates']['INR']
        return usd_value * inr_rate
    except requests.exceptions.RequestException as e:
        print(f"Error fetching USD to INR exchange rate: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing exchange rate data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during USD to INR conversion: {e}")
        return None


class CryptoPricePredictor:
    def __init__(self, models_path=OUTPUT_MODELS_PATH):
        self.models_path = models_path
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.symbols_map = {}
        self.binance_base_url = "https://api.binance.com/api/v3/klines"

        print(f"--- Initializing CryptoPricePredictor from {models_path} ---")
        self._load_all_assets()
        print("--- Predictor Initialized ---")

    def _load_all_assets(self):
        """Loads all saved models and scalers for all coins and timeframes."""
        model_files = [f for f in os.listdir(self.models_path) if f.endswith('_lstm_model.h5')]

        if not model_files:
            raise FileNotFoundError(f"No models found in {self.models_path}. Please run trainer.py first.")

        # Load coin symbols from symbols.json to create Binance-compatible symbols
        try:
            # Assuming symbols.json is in Milestone1 subdirectory from project_root
            symbols_json_path = os.path.join(project_root, 'Milestone1', 'symbols.json')
            with open(symbols_json_path, 'r', encoding='utf-8') as f:
                coins_config = json.load(f)
            for coin in coins_config:
                # Binance pairs are typically like "BTCUSDT" for USDT equivalent
                # This assumes your models were trained on INR data, and Binance provides USDT.
                binance_symbol = coin["symbol"].replace('-INR', 'USDT').upper()
                self.symbols_map[coin["name"].lower()] = binance_symbol
                print(f"Mapping {coin['name']} (from {coin['symbol']}) to Binance symbol: {binance_symbol}")
        except FileNotFoundError:
            print("Warning: symbols.json not found. Predictions might fail without proper symbol mapping.")
        except Exception as e:
            print(f"Error loading symbols.json for symbol mapping: {e}")

        for f_name in model_files:
            parts = f_name.split('_')
            coin_name = parts[0]
            data_type = parts[1]

            key = (coin_name, data_type)
            model_path = os.path.join(self.models_path, f_name)
            scaler_X_path = os.path.join(self.models_path, f'{coin_name}_{data_type}_scaler_X.joblib')
            scaler_y_path = os.path.join(self.models_path, f'{coin_name}_{data_type}_scaler_y.joblib')

            try:
                self.models[key] = load_model(model_path)
                self.scalers_X[key] = joblib.load(scaler_X_path)
                self.scalers_y[key] = joblib.load(scaler_y_path)
                
                # --- DEBUGGING PRINT for Scaler Y ---
                if hasattr(self.scalers_y[key], 'data_min_') and hasattr(self.scalers_y[key], 'data_max_'):
                    print(f"DEBUG: Scaler_y for {key[0].capitalize()} ({key[1]}) fitted on range: "
                          f"Min={self.scalers_y[key].data_min_[0]:.2f}, Max={self.scalers_y[key].data_max_[0]:.2f}")
                # --- END DEBUGGING PRINT ---

                print(f"Loaded model and scalers for {coin_name.capitalize()} ({data_type}).")
            except Exception as e:
                print(f"Error loading assets for {coin_name.capitalize()} ({data_type}): {e}")
                if key in self.models: del self.models[key]
                if key in self.scalers_X: del self.scalers_X[key]
                if key in self.scalers_y: del self.scalers_y[key]


    def _fetch_live_klines(self, symbol, interval, limit):
        """Fetches live klines data from Binance."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        try:
            response = requests.get(self.binance_base_url, params=params)
            response.raise_for_status()
            klines = response.json()
            return klines
        except requests.exceptions.RequestException as e:
            print(f"Error fetching live data for {symbol} ({interval}): {e}")
            return None


    def _prepare_live_data(self, klines_data, model_key):
        """
        Processes live klines data to create features, scales it, and reshapes for LSTM.
        Must match trainer.py's feature engineering exactly.
        """
        if not klines_data:
            return None, None

        df = pd.DataFrame(klines_data, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
            'Taker buy quote asset volume', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])

        # --- Replicate Feature Engineering from trainer.py EXACTLY ---
        df['price_lag_1'] = df['Close'].shift(1)
        df['price_lag_7'] = df['Close'].shift(7)
        df['volume_lag_1'] = df['Volume'].shift(1)
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
        df['volatility_7'] = df['Close'].rolling(window=7).std()
        df['daily_return'] = df['Close'].pct_change()

        df['RSI'] = RSIIndicator(close=df['Close'], window=14, fillna=False).rsi()
        macd_indicator = MACDIndicator(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Diff'] = macd_indicator.macd_diff()

        df['ATR'] = ATRIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=False).average_true_range()
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'], fillna=False).on_balance_volume()

        stoch_indicator = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3, fillna=False)
        df['STOCH_K'] = stoch_indicator.stoch()
        df['STOCH_D'] = stoch_indicator.stoch_signal()

        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        # --- End Feature Engineering ---

        features_to_exclude = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
            'Taker buy quote asset volume', 'Ignore', 'target'
        ]
        features_list = [col for col in df.columns if col not in features_to_exclude]
        
        X_live_raw = df[features_list]

        timesteps = COMMON_TIMESTEPS_DAILY if model_key[1] == 'daily' else COMMON_TIMESTEPS_HOURLY
        
        X_live_raw = X_live_raw.dropna().copy() # Explicitly create a copy after dropping NaNs to avoid SettingWithCopyWarning

        # --- DEBUGGING PRINT for features_list and X_live_raw shape ---
        print(f"DEBUG: Features list for {model_key[0].capitalize()} ({model_key[1]}): {features_list}")
        print(f"DEBUG: X_live_raw shape after dropna: {X_live_raw.shape}")
        # --- END DEBUGGING PRINT ---

        if len(X_live_raw) < timesteps:
            print(f"Error: Not enough valid live data ({len(X_live_raw)} rows) after dropping NaNs to form a sequence of {timesteps} timesteps.")
            return None, None
        
        X_sequence = X_live_raw.iloc[-timesteps:].values

        # --- DEBUGGING PRINT for X_sequence (first row and last row) ---
        print(f"DEBUG: X_sequence (unscaled) for {model_key[0].capitalize()} ({model_key[1]}) - first row of sequence:\n{X_sequence[0]}")
        print(f"DEBUG: X_sequence (unscaled) for {model_key[0].capitalize()} ({model_key[1]}) - last row of sequence:\n{X_sequence[-1]}")
        # --- END DEBUGGING PRINT ---

        scaler_X = self.scalers_X.get(model_key)
        if scaler_X is None:
            print(f"Error: Scaler_X not found for {model_key[0].capitalize()} ({model_key[1]}).")
            return None, None
        
        X_scaled_sequence = scaler_X.transform(X_sequence)
        
        # --- DEBUGGING PRINT for X_scaled_sequence (first row and last row) ---
        print(f"DEBUG: X_scaled_sequence (scaled) for {model_key[0].capitalize()} ({model_key[1]}) - first row of sequence:\n{X_scaled_sequence[0]}")
        print(f"DEBUG: X_scaled_sequence (scaled) for {model_key[0].capitalize()} ({model_key[1]}) - last row of sequence:\n{X_scaled_sequence[-1]}")
        # --- END DEBUGGING PRINT ---

        n_features = X_scaled_sequence.shape[1]
        X_reshaped = X_scaled_sequence.reshape(1, timesteps, n_features)

        # Get the current market price (last close price in the fetched data after dropping NaNs)
        # This will be the USD price from Binance for USDT pairs
        current_market_price_usd = df['Close'].loc[X_live_raw.index[-1]]

        return X_reshaped, current_market_price_usd


    def predict_next_price(self, coin_name_base, data_type):
        """
        Fetches live data, processes it, and predicts the next price for a given coin and timeframe.
        coin_name_base: e.g., 'bitcoin' (lowercase, matches filename prefix)
        data_type: 'daily' or 'hourly'
        """
        model_key = (coin_name_base.lower(), data_type)
        if model_key not in self.models:
            print(f"Error: Model for {model_key[0].capitalize()} ({data_type}) not loaded or does not exist.")
            return None, None, None

        model = self.models.get(model_key)
        scaler_y = self.scalers_y.get(model_key)
        
        binance_symbol = self.symbols_map.get(coin_name_base.lower())
        if not binance_symbol:
            print(f"Error: Binance symbol not mapped for {coin_name_base}. Cannot fetch live data.")
            return None, None, None

        interval_map = {
            'daily': '1d',
            'hourly': '1h'
        }
        binance_interval = interval_map.get(data_type)
        if not binance_interval:
            print(f"Error: Invalid data_type '{data_type}'. Must be 'daily' or 'hourly'.")
            return None, None, None
        
        timesteps = COMMON_TIMESTEPS_DAILY if data_type == 'daily' else COMMON_TIMESTEPS_HOURLY
        fetch_limit = timesteps + 30 + 10 # Ensure enough data for all features and timesteps

        print(f"\n--- Fetching live data for {coin_name_base.capitalize()} ({data_type}) ---")
        klines = self._fetch_live_klines(binance_symbol, binance_interval, fetch_limit)
        if klines is None:
            return None, None, None

        X_predict, current_market_price_usd = self._prepare_live_data(klines, model_key)
        if X_predict is None:
            return None, None, None

        print(f"Making prediction for {coin_name_base.capitalize()} ({data_type})...")
        predicted_scaled = model.predict(X_predict)
        
        # --- DEBUGGING PRINT for Predicted Scaled Value ---
        print(f"DEBUG: Model predicted scaled value: {predicted_scaled[0][0]:.4f}")
        # --- END DEBUGGING PRINT ---

        # The scaler_y was fitted on INR values from your historical CSVs, so inverse_transform will yield INR.
        predicted_price_inr = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()[0]

        return current_market_price_usd, predicted_price_inr, binance_symbol


def main():
    load_dotenv()

    predictor = CryptoPricePredictor()

    if not predictor.models:
        print("No models were loaded. Exiting.")
        return

    try:
        symbols_json_path = os.path.join(project_root, 'Milestone1', 'symbols.json')
        with open(symbols_json_path, "r", encoding="utf-8") as f:
            coins_to_predict = json.load(f)
    except FileNotFoundError:
        print("Error: Missing symbols.json file. Cannot determine coins for live prediction.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid symbols.json format: {e}")
        return

    data_types = ['daily', 'hourly']

    print("\n--- Starting Live Price Prediction ---")
    for coin_data in coins_to_predict:
        coin_name = coin_data["name"].lower()
        for data_type in data_types:
            current_price_usd_from_binance, predicted_price_inr_from_model, binance_symbol = predictor.predict_next_price(coin_name, data_type)
            
            if current_price_usd_from_binance is not None and predicted_price_inr_from_model is not None:
                current_price_inr = convert_usd_to_inr(current_price_usd_from_binance)
                predicted_price_usd = convert_inr_to_usd(predicted_price_inr_from_model)

                print(f"\n📈 {binance_symbol} ({data_type.capitalize()}):")
                print(f"   Current Market Price: {current_price_usd_from_binance:.2f} USD " + (f"({current_price_inr:.2f} INR)" if current_price_inr is not None else "(INR conversion failed)"))
                print(f"   Predicted Next Period Price: {predicted_price_usd:.2f} USD " + (f"({predicted_price_inr_from_model:.2f} INR)" if predicted_price_inr_from_model is not None else "(INR conversion failed)"))
            else:
                print(f"\n⚠️ Could not get prediction for {coin_name.capitalize()} ({data_type}). See errors above.")
    
    print("\n--- Live Prediction Complete ---")


if __name__ == "__main__":
    main()