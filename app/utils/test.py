# utils/test.py
import joblib
import pandas as pd
import numpy as np
import requests
import pvlib
from .train import add_features, physics_predict  # import from your existing file
import os 

# Project root ka path nikaalo
MODEL_PATH = r"C:\Users\aayri\OneDrive\Desktop\Solar_power_generation\Hackovate-APIFetched\app\utils\base_model.joblib"


def fetch_openweather_with_pvlib(lat, lon, api_key):
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&units=metric&appid={api_key}"
    )
    r = requests.get(url, timeout=30)
    data = r.json()

    if "list" not in data:
        raise RuntimeError(f"OpenWeather API error: {data}")

    # Step 1: Build dataframe (timestamps in UTC)
    records = []
    for h in data["list"]:
        dt = pd.to_datetime(h["dt"], unit="s", utc=True)
        records.append({
            "datetime": dt,
            "temp": h["main"].get("temp"),
            "humidity": h["main"].get("humidity"),
            "wind_speed": h["wind"].get("speed"),
            "cloud_cover": h["clouds"].get("all"),
        })

    df = pd.DataFrame(records).set_index("datetime")

    # Step 2: Resample to 1-hourly in UTC
    df = df.resample("1h").interpolate()

    # Step 3: Clearsky model (UTC index)
    site = pvlib.location.Location(lat, lon, tz="UTC")
    cs = site.get_clearsky(df.index)

    # Step 4: Cloud adjustment
    cloud_factor = 1 - (df["cloud_cover"] / 100) * 0.75
    df["ghi"] = (cs["ghi"] * cloud_factor).fillna(0)
    df["dni"] = (cs["dni"] * cloud_factor).fillna(0)
    df["dhi"] = (cs["dhi"] * cloud_factor).fillna(0)

    # Step 5: KT calculation
    df["kt"] = df["ghi"] / (df["dni"] + df["dhi"]).replace(0, np.nan)

    # Step 6: Convert to IST
    df = df.tz_convert("Asia/Kolkata")

    return df


def get_solar_forecast(lat, lon, area, tilt, azimuth, api_key):
    # ✅ Step 1: Current config prediction
    df = fetch_openweather_with_pvlib(lat, lon, api_key)
    df_current = add_features(df.copy(), lat, lon, tilt=tilt, azimuth=azimuth)
    df_current = physics_predict(df_current, area=area)

    saved = joblib.load(MODEL_PATH)
    model, features = saved["model"], saved["features"]

    df_current["predicted_kwh"] = model.predict(df_current[features].fillna(0))
    df_current["predicted_kwh"] = df_current["predicted_kwh"].clip(lower=0)

    hourly = df_current.reset_index()[["datetime", "predicted_kwh"]].to_dict(orient="records")
    daily = df_current["predicted_kwh"].resample("D").sum().reset_index()
    daily = daily.rename(columns={"predicted_kwh": "daily_kwh"}).to_dict(orient="records")
    total = round(float(df_current["predicted_kwh"].sum()), 2)

    # ✅ Step 2: Brute-force optimization (like your training code)
    best_output, best_tilt, best_azimuth = -1, tilt, azimuth
    base_df = df.copy()
    for t in range(0, 61, 10):        # tilt sweep every 5°
        for a in range(90, 271, 30): # azimuth sweep every 15° (East→West)
            df_tmp = add_features(base_df, lat, lon, tilt=t, azimuth=a)
            df_tmp = physics_predict(df_tmp, area=area)
            pred_tmp = model.predict(df_tmp[features].fillna(0)).clip(min=0).sum()
            if pred_tmp > best_output:
                best_output, best_tilt, best_azimuth = pred_tmp, t, a

    optimal_total = round(float(best_output), 2)
    gain = round(((optimal_total - total) / total) * 100, 2) if total > 0 else 0
    

    recommendation = {
        "current_tilt": tilt,
        "current_azimuth": azimuth,
        "optimal_tilt": best_tilt,
        "optimal_azimuth": best_azimuth,
        "current_total": total,
        "optimal_total": optimal_total,
        "potential_gain_percent": gain
    }
    
    latest_weather = df.iloc[0]  # first timestamp ka data (current)
    weather_snapshot = {
        "temperature": round(float(latest_weather["temp"]), 2),
        "humidity": round(float(latest_weather["humidity"]), 2),
        "wind_speed": round(float(latest_weather["wind_speed"]), 2),
        "cloud_cover": round(float(latest_weather["cloud_cover"]), 2),
        "ghi": round(float(latest_weather["ghi"]), 2),
        "dni": round(float(latest_weather["dni"]), 2),
        "dhi": round(float(latest_weather["dhi"]), 2),
        "kt": round(float(latest_weather["kt"]), 3)
    }

    return {
        "hourly": hourly,
        "daily": daily,
        "total": total,
        "recommendation": recommendation,
        "weather": weather_snapshot
    }
