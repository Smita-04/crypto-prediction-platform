import csv
import json
import os
import time
from datetime import datetime
import requests
from dotenv import load_dotenv
load_dotenv() 

class CryptoDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://data-api.coindesk.com/index/cc/v1/historical"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=UTF-8",
            "User-Agent": "CryptoPriceCollector/2.0"
        }

        # Setup directories for saving data
        root_dir = os.getenv("DATA_DIRECTORY", "data")
        self.daily_folder = os.path.join(root_dir, "daily_data")
        self.hourly_folder = os.path.join(root_dir, "hourly_data")
        os.makedirs(self.daily_folder, exist_ok=True)
        os.makedirs(self.hourly_folder, exist_ok=True)

        print("🚀 CryptoDataFetcher initialized successfully!")
        print(f"📂 Daily folder: {self.daily_folder}")
        print(f"📂 Hourly folder: {self.hourly_folder}")

    def get_historical_data(self, symbol, timeframe, limit, delay):
        """Fetch complete historical OHLCV data by batching requests backward."""
        results = []
        end_timestamp = int(datetime.now().timestamp())
        endpoint = f"{self.base_url}/{timeframe}"
        batch = 0

        print(f"🔍 Collecting {timeframe.upper()} data for {symbol}...")

        while True:
            batch += 1
            params = {
                "market": "cadli",
                "instrument": symbol,
                "limit": limit,
                "aggregate": 1,
                "fill": "true",
                "apply_mapping": "true",
                "response_format": "JSON",
                "to_ts": end_timestamp,
                "api_key": self.api_key
            }
            try:
                print(f"📦 Request #{batch} | Fetching data up to: {end_timestamp}")
                response = requests.get(endpoint, params=params, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.json().get("Data", [])

                if not data:
                    print("✅ No more data found — reached the earliest record available.")
                    break

                # Keep only rows with valid OHLCV values
                clean_rows = [
                    row for row in data
                    if all(float(row.get(field, 0)) > 1 for field in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
                ]
                results.extend(clean_rows)
                print(f"📊 Batch collected: {len(clean_rows)} valid rows (Total so far: {len(results)})")

                if len(data) < limit:
                    print("📉 Reached the first available data point.")
                    break

                end_timestamp = min(row["TIMESTAMP"] for row in data) - 1
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                print(f"⚠️ API error: {e}")
                break
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                break

        results.sort(key=lambda x: x["TIMESTAMP"])
        return results

    @staticmethod
    def format_time(ts, timeframe):
        """Convert timestamp to human-readable format."""
        dt = datetime.utcfromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M" if timeframe == "hours" else "%Y-%m-%d")

    def write_to_csv(self, data, symbol, coin_name, timeframe):
        if not data:
            print(f"❌ No {timeframe} data to save for {coin_name}")
            return False

        folder = self.hourly_folder if timeframe == "hours" else self.daily_folder
        file_path = os.path.join(folder, f"{coin_name}.csv")

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Instrument", "Open", "High", "Low", "Close", "Volume"])

                for row in data:
                    writer.writerow([
                        self.format_time(row["TIMESTAMP"], timeframe),
                        row.get("INSTRUMENT", ""),
                        float(row.get("OPEN", 0)),
                        float(row.get("HIGH", 0)),
                        float(row.get("LOW", 0)),
                        float(row.get("CLOSE", 0)),
                        float(row.get("VOLUME", 0)),
                    ])
            print(f"💾 {coin_name} - {timeframe.capitalize()} data saved to {file_path}")
            return True

        except Exception as e:
            print(f"⚠️ Failed to save {coin_name} data: {e}")
            return False

    def run_collection(self, coins, batch_limit, delay):
        print(f"📢 Starting data collection for {len(coins)} coins.")
        success = 0

        for i, coin in enumerate(coins, 1):
            name, symbol = coin["name"], coin["symbol"]
            print(f"\n🔄 [{i}/{len(coins)}] Processing: {name}")

            daily = self.get_historical_data(symbol, "days", batch_limit, delay)
            self.write_to_csv(daily, symbol, name, "days")

            time.sleep(1)

            hourly = self.get_historical_data(symbol, "hours", batch_limit, delay)
            if self.write_to_csv(hourly, symbol, name, "hours"):
                success += 1

            if i < len(coins):
                time.sleep(delay)

        print(f"\n✅ Completed! Successfully collected data for {success}/{len(coins)} coins.")
        return success


def main():
    load_dotenv()

    api_key = os.getenv("COINDESK_API_KEY")
    batch_limit = int(os.getenv("BATCH_LIMIT", "2000"))
    delay = int(os.getenv("REQUEST_DELAY", "1"))

    if not api_key:
        print("⚠️ Please set your COINDESK_API_KEY in the .env file.")
        return

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "symbols.json"), "r", encoding="utf-8") as f:
            coins = json.load(f)
    except FileNotFoundError:
        print("⚠️ Missing symbols.json file.")
        return
    except json.JSONDecodeError as e:
        print(f"⚠️ Invalid symbols.json format: {e}")
        return

    for coin in coins:
        if not coin["symbol"].endswith("-INR"):
            print(f"⚠️ {coin['symbol']} may not be an INR pair!")

    collector = CryptoDataFetcher(api_key)
    start = time.time()
    collector.run_collection(coins, batch_limit, delay)
    duration = (time.time() - start) / 60
    print(f"⏱️ Total runtime: {duration:.2f} minutes")


if __name__ == "__main__":
    main()