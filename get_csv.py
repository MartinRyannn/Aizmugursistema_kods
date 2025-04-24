import os
import requests
import pandas as pd
import time

ec2_ip = "13.60.206.187"
csv_url = f"http://{ec2_ip}:8000/resampled_data.csv"

local_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_resampled_data.csv")

def fetch_csv():
    try:
        response = requests.get(csv_url)

        if response.status_code == 200:
            with open(local_file_path, 'wb') as file:
                file.write(response.content)
            
            df = pd.read_csv(local_file_path)

        else:
            pass 
    except Exception:
        pass

while True:
    fetch_csv()
    time.sleep(10)
