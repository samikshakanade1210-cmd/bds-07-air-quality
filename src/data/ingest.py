import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946}
}

def fetch_live_sensor_fusion_data():
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY missing in .env file!")

    records = []
    for city, coords in CITIES.items():
        # Pollution API call
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={coords['lat']}&lon={coords['lon']}&appid={API_KEY}"
        # Weather API call
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={coords['lat']}&lon={coords['lon']}&appid={API_KEY}"
        
        air_res = requests.get(air_url).json()
        weather_res = requests.get(weather_url).json()

        if "list" in air_res and "main" in weather_res:
            air = air_res['list'][0]
            row = {
                "timestamp": datetime.fromtimestamp(air['dt']),
                "city": city,
                "lat": coords['lat'],
                "lon": coords['lon'],
                "pm2_5": air['components']['pm2_5'],
                "pm10": air['components']['pm10'],
                "no2": air['components']['no2'],
                "so2": air['components']['so2'],
                "co": air['components']['co'],
                "temp_celsius": round(weather_res['main']['temp'] - 273.15, 2),
                "humidity": weather_res['main']['humidity'],
                "pressure": weather_res['main']['pressure'],
                "wind_speed": weather_res['wind']['speed'],
                "wind_deg": weather_res['wind'].get('deg', 0)
            }
            records.append(row)

    df = pd.DataFrame(records)
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/raw_sensor_fusion.csv"
    df.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
    print(f"[SUCCESS] Ingested {len(df)} records into {out_path}")

if __name__ == "__main__":
    fetch_live_sensor_fusion_data()