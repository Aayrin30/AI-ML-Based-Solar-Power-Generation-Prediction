# AI/ML-Based Solar Power Generation Prediction 🌞

A **Solar Power Prediction Platform** leveraging AI/ML to provide accurate, site-specific solar power output predictions based on environmental and physical parameters.

---

## 🚀 Problem Statement
Solar energy is a promising renewable source, but actual power generation depends on multiple dynamic factors such as location, sunlight, panel orientation, shading, and weather. Traditional estimation methods often fail to capture real-world variations, leading to overestimation or underutilization.

This platform predicts solar power output, suggests optimal panel placement, and provides actionable insights through a user-friendly dashboard.

---

## 🛠 Implementation
1. **Data Sourcing**
   - Historical solar data from **NASA API** for training the model.
   - User inputs: latitude, longitude, panel surface area, tilt & azimuth angles.
   - Current weather data fetched via **OpenWeather API** (temperature, humidity, wind speed, cloud cover).

2. **Solar Power Calculation**
   - Used **PVlib** library to calculate:
     - **DHI, GHI, DNI** (irradiance components)
     - **Hourly kWh generation**
   - Generated predictions for:
     - Daily power generation
     - Hourly generation for selected date
     - 5-day forecast bar charts

3. **Optimal Configuration & Insights**
   - Compares current vs. **optimal tilt & azimuth angles**
   - Shows improvement as a **percentage gain**
   - Provides actionable recommendations for panel placement

4. **Frontend/UI Features**
   - **Interactive Dashboard** for input and visualization
   - Graphs:
     - Hourly kWh generation
     - 5-Days output
     - Current vs Optimal configuration
   - **Automatic location fetching** using **Jio API** for autocomplete and reverse geocoding
   - **Multi-lingual support**
   - **Chatbot assistance** for help

5. **Reporting**
   - Export predicted data as **CSV or PDF**
   - Reports include daily, weekly, monthly generation and optimization suggestions

---

## 💡 Key Features
- AI/ML-powered predictions
- Multi-factor input: location, panel orientation, weather, seasonal variations
- Visual insights: line charts, bar charts
- Automatic optimal configuration recommendations
- Multi-lingual interface
- Chatbot for user assistance

---

## 📈 Example Graphs
*(Include screenshots of your dashboard graphs here)*

---

## 🛠 Tech Stack
- **Backend:** Python, PVlib, OpenWeather API, NASA API
- **Frontend:** HTML, CSS, JavaScript, Chart.js (or your dashboard framework)
- **Other:** Jio API for location autocomplete, Multi-lingual support, Chatbot integration

---


## 📂 Usage
1. Clone the repository:
   ```bash
   git clone "https://github.com/Aayrin30/AI-ML-Based-Solar-Power-Generation-Prediction.git"
   ```
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
   ```
3.Run the application:
   ```bash
python manage.py runserver
   ```
4.Open the dashboard in your browser and start predicting solar power output.
