import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json # ADDED: For saving training history
import requests # ADDED: For INR to USD conversion

# --- CORRECTED TA IMPORTS (with additional indicators) ---
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD as MACDIndicator
from ta.volatility import AverageTrueRange as ATRIndicator
from ta.volume import OnBalanceVolumeIndicator
# --- END CORRECTED TA IMPORTS ---

# TensorFlow/Keras Imports for LSTM and GRU
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU # ADDED GRU layer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import AdamW # ADDED AdamW optimizer

# Ensure TensorFlow does not allocate all GPU memory at once (if using GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Robust Path Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

BASE_DATA_ROOT = os.path.join(project_root, 'data')

OUTPUT_PLOTS_PATH = os.path.join(project_root, 'plots')
OUTPUT_MODELS_PATH = os.path.join(project_root, 'models')

os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)
os.makedirs(OUTPUT_MODELS_PATH, exist_ok=True)

print(f"Output plots will be saved in: {OUTPUT_PLOTS_PATH}")
print(f"Output models will be saved in: {OUTPUT_MODELS_PATH}")
print(f"Attempting to scan for data in: {BASE_DATA_ROOT}")

# --- GLOBAL DATA SPLIT RATIOS ---
GLOBAL_TRAIN_RATIO = 0.7
GLOBAL_VAL_RATIO = 0.15
# --- END GLOBAL DATA SPLIT RATIOS ---

# --- Helper Function for LSTM Data Reshaping ---
def create_lstm_dataset(X, y, timesteps):
    Xs, ys = [], []
    y_np = y.flatten() if isinstance(y, np.ndarray) else y.values.flatten()

    for i in range(len(X) - timesteps):
        v = X.iloc[i:(i + timesteps)].values
        Xs.append(v)
        ys.append(y_np[i + timesteps])
    
    if not Xs or not ys:
        return np.array([]), np.array([])
        
    return np.array(Xs), np.array(ys)

# --- Function to Build Hybrid BiLSTM + GRU Model (UPDATED) ---
def build_lstm_model(timesteps, n_features): # Renamed to reflect its purpose from now on
    model = Sequential([
        Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(timesteps, n_features)), # Tuned units
        Dropout(0.2),
        GRU(units=64, return_sequences=False), # Hybrid layer
        Dropout(0.2),
        Dense(units=1)
    ])
    
    # Using AdamW optimizer (UPDATED)
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# --- Helper function for MAPE ---
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero, replace 0s in y_true with a small epsilon
    # or handle cases where y_true is 0 based on context.
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Helper function to convert INR to USD (ADDED) ---
def convert_inr_to_usd(inr_value):
    """Fetches current INR to USD exchange rate and converts the given INR value."""
    if inr_value is None:
        return None
    try:
        response = requests.get("https://open.er-api.com/v6/latest/INR")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
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

# --- Main Data Preparation Function ---
def prepare_data_for_modeling(coin_csv_filename, data_type='daily', timesteps=30):
    coin_name_base = coin_csv_filename.replace('.csv', '')
    print(f"\n--- Preparing Data for {coin_name_base} ({data_type}) Modeling ---")
    
    current_data_path = os.path.join(BASE_DATA_ROOT, f'{data_type}_data')
    file_path = os.path.join(current_data_path, coin_csv_filename)

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {coin_csv_filename}")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}. Skipping.")
        return None, None, None, None, None, None, None, None

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in {coin_csv_filename}. Skipping.")
        return None, None, None, None, None, None, None, None

    # --- Part 2: Advanced Feature Engineering ---
    print("--- Starting Advanced Feature Engineering ---")

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

    df['target'] = df['Close'].shift(-1) 

    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows with NaN values after feature engineering.")

    if df.empty:
        print(f"Not enough data after cleaning and feature engineering for {coin_name_base}. Skipping further processing.")
        return None, None, None, None, None, None, None, None

    features_list = [
        col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Instrument', 'target']
    ]
    
    X = df[features_list]
    y = df['target']
    
    print(f"Engineered Features: {features_list}")
    print(f"Data shape after feature engineering: {X.shape}")

    # --- Part 3: Data Scaling ---
    print("--- Scaling Features (MinMaxScaler) ---")
    scaler_X = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # --- Part 4: Data Splitting (Chronological Train/Validation/Test) ---
    print("--- Splitting Data into Training, Validation, and Testing Sets ---")

    total_rows = len(X_scaled)
    train_size = int(total_rows * GLOBAL_TRAIN_RATIO)
    val_size = int(total_rows * GLOBAL_VAL_RATIO)
    
    if (train_size < timesteps + 1) or (val_size < timesteps + 1) or (total_rows - train_size - val_size < timesteps + 1):
        print(f"Not enough data to create valid LSTM sequences (min {timesteps+1} samples per set) for {coin_name_base} ({data_type}) with {timesteps} timesteps. Skipping.")
        return None, None, None, None, None, None, None, None

    X_train_raw = X_scaled.iloc[:train_size]
    y_train_raw = y_scaled[:train_size]

    X_val_raw = X_scaled.iloc[train_size : train_size + val_size]
    y_val_raw = y_scaled[train_size : train_size + val_size]

    X_test_raw = X_scaled.iloc[train_size + val_size : ]
    y_test_raw = y_scaled[train_size + val_size : ]

    # --- Part 5: LSTM Data Reshaping ---
    print(f"--- Reshaping data for LSTM (timesteps={timesteps}) ---")
    X_train_reshaped, y_train_reshaped = create_lstm_dataset(X_train_raw, y_train_raw, timesteps)
    X_val_reshaped, y_val_reshaped = create_lstm_dataset(X_val_raw, y_val_raw, timesteps)
    X_test_reshaped, y_test_reshaped = create_lstm_dataset(X_test_raw, y_test_raw, timesteps)

    if X_train_reshaped.size == 0 or X_val_reshaped.size == 0 or X_test_reshaped.size == 0:
        print(f"Warning: One or more LSTM datasets are empty after reshaping for {coin_name_base} ({data_type}). Skipping.")
        return None, None, None, None, None, None, None, None

    print(f"Train Set Shape (LSTM): X={X_train_reshaped.shape}, y={y_train_reshaped.shape}")
    print(f"Validation Set Shape (LSTM): X={X_val_reshaped.shape}, y={y_val_reshaped.shape}")
    print(f"Test Set Shape (LSTM): X={X_test_reshaped.shape}, y={y_test_reshaped.shape}")
    print(f"Data preparation for {coin_name_base} ({data_type}) complete.")

    return X_train_reshaped, y_train_reshaped, X_val_reshaped, y_val_reshaped, \
           X_test_reshaped, y_test_reshaped, scaler_X, scaler_y

# --- Main execution block ---
if __name__ == "__main__":
    if not os.path.exists(BASE_DATA_ROOT):
        print(f"Error: The data root directory does not exist at '{BASE_DATA_ROOT}'.")
        print("Please ensure your 'data' folder is directly under your project root (same level as Milestone1, plots, models).")
        print(f"Project root identified as: '{project_root}'")
        print(f"Contents of project root '{project_root}': {os.listdir(project_root) if os.path.exists(project_root) else 'Not found'}")
        exit()
    
    if not os.path.isdir(BASE_DATA_ROOT):
        print(f"Error: '{BASE_DATA_ROOT}' is not a directory.")
        exit()

    data_types = ['daily', 'hourly']
    common_timesteps_daily = 30
    common_timesteps_hourly = 24

    for data_type in data_types:
        current_data_dir = os.path.join(BASE_DATA_ROOT, f'{data_type}_data')
        if not os.path.exists(current_data_dir):
            print(f"\nWarning: '{data_type}_data' directory not found at '{current_data_dir}'. Skipping {data_type} data.")
            continue

        all_coin_files = [f for f in os.listdir(current_data_dir) if f.endswith('.csv')]

        if not all_coin_files:
            print(f"\nNo CSV files found in the {data_type}_data directory '{current_data_dir}'. Skipping {data_type} data.")
            continue

        print(f"\n--- Processing {data_type.upper()} Data (Found {len(all_coin_files)} coins) ---")
        
        timesteps = common_timesteps_daily if data_type == 'daily' else common_timesteps_hourly

        for coin_file in all_coin_files:
            coin_name = coin_file.replace('.csv', '').lower()
            
            X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = \
                prepare_data_for_modeling(coin_file, data_type=data_type, timesteps=timesteps)

            if X_train is None or X_train.size == 0:
                continue

            n_features = X_train.shape[2]

            print(f"\nBuilding and Training Hybrid Bi-LSTM + GRU model for {coin_name} ({data_type})...")
            lstm_model = build_lstm_model(timesteps, n_features) # Using the updated build_lstm_model
            
            # --- Callbacks for Training ---
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            model_checkpoint_filepath = os.path.join(OUTPUT_MODELS_PATH, f'{coin_name}_{data_type}_best_model_checkpoint.h5')
            model_checkpoint = ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', save_best_only=True, verbose=0)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

            callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

            history = lstm_model.fit(
                X_train, y_train,
                epochs=150, # Hyperparameter: Epochs
                batch_size=32, # Hyperparameter: Batch size
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=0
            )
            print(f"Hybrid Bi-LSTM + GRU Model Training for {coin_name} ({data_type}) Complete. Epochs run: {len(history.history['loss'])}")

            # --- Save training history to JSON (ADDED) ---
            history_filename = os.path.join(OUTPUT_MODELS_PATH, f'{coin_name}_{data_type}_training_history.json')
            try:
                with open(history_filename, 'w') as f:
                    json.dump(history.history, f, indent=4)
                print(f"Training history saved to {history_filename}")
            except Exception as e:
                print(f"Error saving training history for {coin_name} ({data_type}): {e}")

            loss_scaled = lstm_model.evaluate(X_test, y_test, verbose=0)

            # --- Make and Display Predictions ---
            print(f"Generating predictions for {coin_name} ({data_type})...")
            test_predictions_scaled = lstm_model.predict(X_test)
            
            test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

            print(f"\n--- Sample Predictions (First 5) for {coin_name} ({data_type}) ---")
            for i in range(min(5, len(test_predictions_original))):
                predicted_inr = test_predictions_original[i]
                actual_inr = y_test_original[i]
                predicted_usd = convert_inr_to_usd(predicted_inr) # Using the new helper
                actual_usd = convert_inr_to_usd(actual_inr) # Using the new helper
                
                print(f"Actual: {actual_inr:.2f} INR " + (f"({actual_usd:.2f} USD)" if actual_usd is not None else "") +
                      f", Predicted: {predicted_inr:.2f} INR " + (f"({predicted_usd:.2f} USD)" if predicted_usd is not None else ""))
            
            # --- Performance Metrics Calculation ---
            mse = mean_squared_error(y_test_original, test_predictions_original)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_original, test_predictions_original)
            r2 = r2_score(y_test_original, test_predictions_original)
            mape = mean_absolute_percentage_error(y_test_original, test_predictions_original)

            print(f"\n--- Evaluation Metrics for {coin_name} ({data_type}) ---")
            print(f"R2 Score: {r2:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


            # --- Directional Accuracy Calculation ---
            temp_file_path_for_plot = os.path.join(current_data_dir, coin_file)
            full_df_for_direction = pd.read_csv(temp_file_path_for_plot, parse_dates=['Date'], index_col='Date')
            full_df_for_direction.sort_index(inplace=True)
            
            full_df_for_direction['target'] = full_df_for_direction['Close'].shift(-1)
            full_df_for_direction.dropna(inplace=True)

            total_cleaned_rows = len(full_df_for_direction)
            train_size_cleaned = int(total_cleaned_rows * GLOBAL_TRAIN_RATIO)
            val_size_cleaned = int(total_cleaned_rows * GLOBAL_VAL_RATIO)
            
            actual_prices_for_direction_start_idx = train_size_cleaned + val_size_cleaned + timesteps -1
            
            if actual_prices_for_direction_start_idx < len(full_df_for_direction):
                current_prices_for_direction = full_df_for_direction['Close'].iloc[actual_prices_for_direction_start_idx : actual_prices_for_direction_start_idx + len(y_test_original)].values
            else:
                current_prices_for_direction = np.array([])
                print("Warning: Not enough 'current' prices to calculate directional accuracy.")

            if len(current_prices_for_direction) == len(y_test_original):
                actual_direction = (y_test_original.flatten() > current_prices_for_direction).astype(int)
                predicted_direction = (test_predictions_original.flatten() > current_prices_for_direction).astype(int)
                
                correct_directions = np.sum(actual_direction == predicted_direction)
                directional_accuracy = correct_directions / len(y_test_original) * 100
                print(f"Directional Accuracy for {coin_name} ({data_type}): {directional_accuracy:.2f}%")
            else:
                print(f"Could not calculate directional accuracy for {coin_name} ({data_type}) due to shape mismatch.")


            plt.figure(figsize=(16, 8))
            
            plot_df = pd.read_csv(temp_file_path_for_plot, parse_dates=['Date'], index_col='Date')
            plot_df.sort_index(inplace=True)

            plot_df['price_lag_1'] = plot_df['Close'].shift(1)
            plot_df['price_lag_7'] = plot_df['Close'].shift(7)
            plot_df['volume_lag_1'] = plot_df['Volume'].shift(1)
            plot_df['SMA_7'] = plot_df['Close'].rolling(window=7).mean()
            plot_df['SMA_30'] = plot_df['Close'].rolling(window=30).mean()
            plot_df['EMA_14'] = plot_df['Close'].ewm(span=14, adjust=False).mean()
            plot_df['volatility_7'] = plot_df['Close'].rolling(window=7).std()
            plot_df['daily_return'] = plot_df['Close'].pct_change()
            plot_df['RSI'] = RSIIndicator(close=plot_df['Close'], window=14, fillna=False).rsi()
            macd_indicator_temp = MACDIndicator(close=plot_df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
            plot_df['MACD'] = macd_indicator_temp.macd()
            plot_df['MACD_Signal'] = macd_indicator_temp.macd_signal()
            plot_df['MACD_Diff'] = macd_indicator_temp.macd_diff()
            plot_df['ATR'] = ATRIndicator(high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], window=14, fillna=False).average_true_range()
            plot_df['OBV'] = OnBalanceVolumeIndicator(close=plot_df['Close'], volume=plot_df['Volume'], fillna=False).on_balance_volume()
            stoch_indicator_temp = StochasticOscillator(high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], window=14, smooth_window=3, fillna=False)
            plot_df['STOCH_K'] = stoch_indicator_temp.stoch()
            plot_df['STOCH_D'] = stoch_indicator_temp.stoch_signal()
            plot_df['day_of_week'] = plot_df.index.dayofweek
            plot_df['day_of_month'] = plot_df.index.day
            plot_df['month'] = plot_df.index.month
            plot_df['year'] = plot_df.index.year
            plot_df['quarter'] = plot_df.index.quarter
            plot_df['target'] = plot_df['Close'].shift(-1)
            plot_df.dropna(inplace=True)


            test_start_index_for_plot = train_size_cleaned + val_size_cleaned + timesteps
            
            if test_start_index_for_plot < len(plot_df.index):
                test_dates = plot_df.index[test_start_index_for_plot : test_start_index_for_plot + len(y_test_original)]
            else:
                test_dates = np.arange(len(y_test_original))
                print("Warning: Not enough dates in plot_df to align with predictions. Plotting against sequence index.")


            if len(test_dates) == len(y_test_original):
                plt.plot(test_dates, y_test_original, label='Actual Price', color='blue', linewidth=1.5)
                plt.plot(test_dates, test_predictions_original, label='Predicted Price', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate()
                plt.xlabel('Date')
            else:
                plt.plot(y_test_original, label='Actual Price', color='blue', linewidth=1.5)
                plt.plot(test_predictions_original, label='Predicted Price', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                plt.xlabel('Time Step (in Test Set Sequence)')
            
            plt.title(f'{coin_name.capitalize()} {data_type.capitalize()} Price Prediction vs Actual (Test Set)')
            plt.ylabel('Price (INR)')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plot_filename = os.path.join(OUTPUT_PLOTS_PATH, f'{coin_name}_{data_type}_predictions_vs_actuals.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Prediction plot saved to {plot_filename}")

            final_model_save_filename = f"{coin_name}_{data_type}_lstm_model.h5"
            final_model_save_path = os.path.join(OUTPUT_MODELS_PATH, final_model_save_filename)
            lstm_model.save(final_model_save_path)
            print(f"Final Model saved to {final_model_save_path}")

            # Save scalers for predictor.py
            scaler_X_path = os.path.join(OUTPUT_MODELS_PATH, f'{coin_name}_{data_type}_scaler_X.joblib')
            scaler_y_path = os.path.join(OUTPUT_MODELS_PATH, f'{coin_name}_{data_type}_scaler_y.joblib')
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)
            print(f"Scalers saved to {scaler_X_path} and {scaler_y_path}")


    print("\n--- All cryptocurrency LSTM models built, trained, and saved. ---")
    print(f"Check the '{os.path.basename(OUTPUT_MODELS_PATH)}' folder for the generated models.")
    print(f"Check the '{os.path.basename(OUTPUT_PLOTS_PATH)}' folder for the plots.")