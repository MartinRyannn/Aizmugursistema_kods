import tpqoa
import pandas as pd
import os
from datetime import datetime
import sys

api = tpqoa.tpqoa(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oanda.cfg'))

instrument = "XAU_USD"
start_date = "2024-09-01"
end_date = "2024-09-30"

date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days

granularities = {
    'M1': '1min',
    'M5': '5min',
    'M10': '10min',
    'M15': '15min',
    'M30': '30min'
}

for granularity, folder in granularities.items():
    os.makedirs(f"data/{folder}", exist_ok=True)

for date in date_range:
    for granularity, folder in granularities.items():
        try:
            data = api.get_history(
                instrument=instrument,
                start=date.date().isoformat() + "T00:00:00",
                end=date.date().isoformat() + "T23:59:59",
                granularity=granularity,
                price='M',
                localize=False
            )

            if data.empty:
                print(f"No data returned for {date.date()} (Granularity: {granularity})")
                continue

            csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/{folder}/XAU_USD_{date.date()}_{granularity}.csv")
            data.to_csv(csv_file_path, index=True)
            print(f"Data for {date.date()} (Granularity: {granularity}) downloaded and saved to {csv_file_path}")

        except Exception as e:
            print(f"Error downloading data for {date.date()} (Granularity: {granularity}): {e}")

print("Data download process completed.")
