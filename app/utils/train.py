# solar_model.py

import requests
import pandas as pd
import numpy as np
import pvlib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

MODEL_PATH = "base_model.joblib"

# ------------------------------
# 1. Fetch NASA data (historical use)
# ------------------------------
def fetch_nasa_data(lat, lon, start="20220101", end="20221231"):
    params = ",".join([
        "T2M",                   # Temperature
        "ALLSKY_SFC_SW_DWN",     # GHI
        "ALLSKY_SFC_SW_DIFF",    # DHI
        "ALLSKY_KT",             # Clearness Index
        "RH2M",                  # Relative Humidity
        "WS2M",                  # Wind Speed
        "CLOUD_AMT"              # Cloud Amount (cloud cover %)
    ])

    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters={params}&community=RE"
        f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )

    r = requests.get(url, timeout=30)
    data = r.json()

    if "properties" not in data or "parameter" not in data["properties"]:
        raise RuntimeError(f"NASA API error: {data.get('messages', data)}")

    parameters = data["properties"]["parameter"]

    t2m = parameters.get("T2M", {})
    ghi_s = parameters.get("ALLSKY_SFC_SW_DWN", {})
    dhi_s = parameters.get("ALLSKY_SFC_SW_DIFF", {})
    kt_s = parameters.get("ALLSKY_KT", {})
    rh2m = parameters.get("RH2M", {})
    ws2m = parameters.get("WS2M", {})
    cloud = parameters.get("CLOUD_AMT", {})

    records = []
    for date, temp in t2m.items():
        dt = pd.to_datetime(date, format="%Y%m%d%H", utc=True)  # ✅ ensure UTC
        records.append({
            "datetime": dt,
            "ghi": ghi_s.get(date),
            "dhi": dhi_s.get(date),
            "dni": np.nan,
            "temp": temp,
            "kt": kt_s.get(date),
            "humidity": rh2m.get(date),
            "wind_speed": ws2m.get(date),
            "cloud_cover": cloud.get(date),
        })

    df = pd.DataFrame(records).set_index("datetime")

    # ✅ Convert UTC → IST
    df = df.tz_convert("Asia/Kolkata")

    return df


def safe_fetch_nasa_data(lat, lon, start, end):
    """
    Generate synthetic clear-sky dataset for forecast (future dates).
    """
    times = pd.date_range(start=pd.to_datetime(start, format="%Y%m%d"),
                          end=pd.to_datetime(end, format="%Y%m%d"),
                          freq="1H", tz="UTC")

    site = pvlib.location.Location(lat, lon)
    cs = site.get_clearsky(times)  # GHI, DNI, DHI (W/m²)

    temp_air = 25 + 7 * np.sin(2 * np.pi * times.dayofyear / 365)
    rh = np.full(len(times), 50)       # Humidity %
    ws = np.full(len(times), 2.0)      # Wind Speed m/s
    cloud = np.full(len(times), 20)    # Cloud cover %

    df = pd.DataFrame({
        "datetime": times,
        "ghi": cs["ghi"],
        "dhi": cs["dhi"],
        "dni": cs["dni"],
        "temp": temp_air,
        "kt": cs["ghi"] / (cs["dni"] + cs["dhi"]).replace(0, np.nan),
        "humidity": rh,
        "wind_speed": ws,
        "cloud_cover": cloud,
    }).set_index("datetime")

    # ✅ Convert UTC → IST
    df = df.tz_convert("Asia/Kolkata")

    return df

# ------------------------------
# 3. Feature Engineering
# ------------------------------
def add_features(df, lat, lon, tilt=15, azimuth=180):
    # ✅ Ensure timezone IST
    if df.index.tz is None:
        df = df.tz_localize("Asia/Kolkata")
    else:
        df = df.tz_convert("Asia/Kolkata")

    # Solar position
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    df["solar_zenith"] = solpos["zenith"]
    df["solar_azimuth"] = solpos["azimuth"]

    # Agar DNI/DHI missing hain → calculate using ERBS
    if "dni" not in df.columns or df["dni"].isnull().all():
        erbs = pvlib.irradiance.erbs(df["ghi"], df["solar_zenith"], df.index)
        df["dni"] = erbs["dni"]
        df["dhi"] = erbs["dhi"]

    # POA irradiance
    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=df["solar_zenith"],
        solar_azimuth=df["solar_azimuth"],
        ghi=df["ghi"],
        dni=df["dni"],
        dhi=df["dhi"]
    )
    df["poa_irradiance"] = total["poa_global"]

    # Cell temperature
    df["T_cell"] = df["temp"] + (45 - 20) * (df["poa_irradiance"] / 800)

    # Time features
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofyear"] = df.index.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Defaults if missing
    df["humidity"] = df.get("humidity", 50).fillna(50)
    df["wind_speed"] = df.get("wind_speed", 2.0).fillna(2.0)
    df["cloud_cover"] = df.get("cloud_cover", 20).fillna(20)

    return df

# ------------------------------
# 4. Physics baseline
# ------------------------------
def physics_predict(df, area=10, eff=0.18, derate=0.9, temp_coeff=-0.004):
    df["eta_adj"] = eff * (1 + temp_coeff * (df["T_cell"] - 25))
    df["eta_adj"] = df["eta_adj"].clip(lower=0)
    df["power_w"] = df["poa_irradiance"] * area * df["eta_adj"] * derate
    df["power_kw"] = df["power_w"] / 1000
    df["energy_kwh"] = df["power_kw"]
    return df


# ------------------------------
# 5. Train Model
# ------------------------------
def train_base_model(lat=28.6139, lon=77.2090,
                     start="20220101", end="20221231",
                     tilt=15, azimuth=180, panel_area=10):

    df = fetch_nasa_data(lat, lon, start, end)
    df = add_features(df, lat, lon, tilt=tilt, azimuth=azimuth)
    df = physics_predict(df, area=panel_area)

    features = [
        "ghi", "dhi", "dni", "temp", "humidity", "wind_speed", "cloud_cover", "kt",
        "solar_zenith", "solar_azimuth", "poa_irradiance", "T_cell",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]

    X = df[features].fillna(0)
    y = df["energy_kwh"]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=500,
        reg_alpha=0.1,   # L1 regularization
        reg_lambda=0.1   # L2 regularization
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test)
    
    # ✅ Error metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"ML Model -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


# ------------------------------
# 6. Predict from User Inputs
# ------------------------------
def predict_from_user_inputs(lat, lon, panel_area=10, tilt=15, azimuth=180,
                             start="20220101", end="20241231",
                             optimize=True):
    start_year, end_year = int(start[:4]), int(end[:4])

    # ------------------ DATA FETCH --------------
    if end_year <= 2024:
        print("✅ Using NASA Historical Data")
        df = fetch_nasa_data(lat, lon, start, end)
    else:
        print("⚠️ Future dates detected → Using Forecast Mode (synthetic clear-sky)")
        df = safe_fetch_nasa_data(lat, lon, start, end)

    # ------------------ FEATURE ENGINEERING ------------------
    df = add_features(df, lat, lon, tilt=tilt, azimuth=azimuth)
    df = physics_predict(df, area=panel_area)

    # ------------------ ML PREDICTION ------------------
    saved = joblib.load(MODEL_PATH)
    model, features = saved["model"], saved["features"]
    df["predicted_kwh"] = model.predict(df[features].fillna(0))
    df["predicted_kwh"] = df["predicted_kwh"].clip(lower=0)     
    # ------------------ AGGREGATES ------------------
    hourly = df[["ghi", "temp", "humidity", "wind_speed", "cloud_cover",
                 "poa_irradiance", "predicted_kwh"]]
    daily = df["predicted_kwh"].resample("D").sum()
    total = df["predicted_kwh"].sum()

    # ------------------ OPTIMIZATION ------------------
    recommendation = None
    if optimize:
        best_output, best_tilt, best_azimuth = -1, tilt, azimuth
        for t in range(0, 61, 15):        # tilt 0–60
            for a in range(90, 271, 30):  # azimuth 90–270
                df_tmp = add_features(df.copy(), lat, lon, tilt=t, azimuth=a)
                df_tmp = physics_predict(df_tmp, area=panel_area)
                pred_tmp = model.predict(df_tmp[features].fillna(0)).sum()
                if pred_tmp > best_output:
                    best_output, best_tilt, best_azimuth = pred_tmp, t, a

        gain = ((best_output - total) / total) * 100 if total > 0 else 0
        recommendation = {
            "current_total_kWh": total,
            "best_total_kWh": best_output,
            "tilt": best_tilt,
            "azimuth": best_azimuth,
            "gain_percent": round(gain, 2)
        }

    return hourly, daily, total, recommendation

# ------------------------------
# Run Example
# ------------------------------
if __name__ == "__main__":
    # Train model once on past data
    train_base_model()

    # Forecast example (future dates)
    hourly, daily, total, rec = predict_from_user_inputs(
        lat=23.0225, lon=72.5714,
        panel_area=10, tilt=20, azimuth=180,
        start="20220101", end="20241231",
    )

    print("\n================= 🌞 SOLAR PREDICTION REPORT =================")
    print("\n---- ⏰ Hourly Predictions (first 24 hrs) ----")
    print(hourly.head(24).to_string(index=True, justify="center"))

    print("\n---- 📅 Daily Totals (kWh) ----")
    for d, val in daily.items():
        print(f"   {d.strftime('%Y-%m-%d')}: {val:.2f} kWh")

    print("\n---- ⚡ Overall Total Energy ----")
    print(f"   Total Energy Generated: {total:.2f} kWh")

    if rec:
        print("\n---- 🔧 Panel Orientation Recommendation ----")
        print(f"   Current Total   : {rec['current_total_kWh']:.2f} kWh")
        print(f"   Best Possible   : {rec['best_total_kWh']:.2f} kWh")
        print(f"   Optimal Tilt    : {rec['tilt']}°")
        print(f"   Optimal Azimuth : {rec['azimuth']}°")
        print(f"   Efficiency Gain : +{rec['gain_percent']:.2f}%")
    print("\n===============================================================\n")