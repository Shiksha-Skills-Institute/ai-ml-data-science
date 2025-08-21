Of course. Here is a complete, runnable Python solution for the **Ahmedabad Air Quality & Traffic Analysis** project.

This solution uses simulated data for demonstration, but the code structure is exactly what a student would use with live APIs. It's broken down into the key phases of the project.

-----

### \#\# Phase 1 & 2: Data Collection, Cleaning, and EDA

This script simulates fetching data, cleans it, merges it, and performs a basic exploratory data analysis (EDA). This code can be run in a Jupyter Notebook.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# --- Phase 1: Simulate Data Collection ---
# In a real project, this part would be replaced with API calls.
print("Simulating data collection...")
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
date_range = pd.date_range(start=start_date, end=end_date, freq='H')

# Simulate Air Quality Data (AQI)
aqi_data = {
    'timestamp': date_range,
    'station': 'Satellite',
    'aqi': np.random.randint(80, 250, size=len(date_range)),
    'pm2_5': np.random.uniform(50, 150, size=len(date_range))
}
aqi_df = pd.DataFrame(aqi_data)

# Simulate Traffic Data (Travel Time in minutes from a key route)
traffic_data = {
    'timestamp': date_range,
    'route': 'SG Highway to Kalupur',
    'travel_time_mins': np.random.randint(25, 75, size=len(date_range))
}
traffic_df = pd.DataFrame(traffic_data)

# Simulate Weather Data
weather_data = {
    'timestamp': date_range,
    'temperature_c': np.random.uniform(28, 42, size=len(date_range)),
    'humidity_percent': np.random.uniform(40, 85, size=len(date_range)),
    'wind_speed_kmh': np.random.uniform(5, 20, size=len(date_range))
}
weather_df = pd.DataFrame(weather_data)

# --- Phase 2: Data Cleaning & Merging ---
print("Cleaning and merging data...")
# Set timestamp as the index for all dataframes
aqi_df.set_index('timestamp', inplace=True)
traffic_df.set_index('timestamp', inplace=True)
weather_df.set_index('timestamp', inplace=True)

# Merge into a single dataframe
df = aqi_df.join(traffic_df).join(weather_df)
df.drop(['station', 'route'], axis=1, inplace=True) # Drop redundant columns

# Check for and handle missing values (if any)
df.fillna(method='ffill', inplace=True)
print("\nMerged Data Head:")
print(df.head())

# --- Phase 2: Exploratory Data Analysis (EDA) ---
print("\nPerforming EDA...")
sns.set_style("whitegrid")

# 1. Plot AQI over time
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['aqi'], label='AQI', color='red')
plt.title('AQI Trend in Ahmedabad (Last 30 Days)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Air Quality Index (AQI)')
plt.legend()
plt.show()

# 2. Plot Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between AQI, Traffic, and Weather', fontsize=16)
plt.show()

# 3. Analyze patterns by hour of the day
df['hour'] = df.index.hour
hourly_avg = df.groupby('hour')[['aqi', 'travel_time_mins']].mean()

fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(hourly_avg.index, hourly_avg['aqi'], 'g-', label='Average AQI')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Average AQI', color='g')
ax1.tick_params('y', colors='g')

ax2 = ax1.twinx()
ax2.plot(hourly_avg.index, hourly_avg['travel_time_mins'], 'b-', label='Average Travel Time')
ax2.set_ylabel('Average Travel Time (mins)', color='b')
ax2.tick_params('y', colors='b')

plt.title('Average AQI and Traffic by Hour of Day', fontsize=16)
fig.tight_layout()
plt.show()

```

-----

### \#\# Phase 3: Model Building & Forecasting

Here, we use the cleaned data to build a predictive model. We'll use **Facebook Prophet**, which is excellent for time-series forecasting with daily and weekly patterns.

**First, install the required library:**
`pip install prophet`

```python
from prophet import Prophet
import pandas as pd # Ensure pandas is imported

# --- Prepare data for Prophet ---
# Prophet requires columns to be named 'ds' (datestamp) and 'y' (value to predict)
prophet_df = df.reset_index().rename(columns={'timestamp': 'ds', 'aqi': 'y'})[['ds', 'y']]

print("Training Prophet model...")
# --- Build and Train the Model ---
# We instantiate a new Prophet object. We can add seasonality components.
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(prophet_df)

# --- Make Future Predictions ---
# Create a dataframe for future timestamps (e.g., next 48 hours)
future = model.make_future_dataframe(periods=48, freq='H')
forecast = model.predict(future)

print("\nForecast Data Head:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# --- Visualize the Forecast ---
print("\nPlotting forecast...")
fig1 = model.plot(forecast, figsize=(15, 7))
plt.title('AQI Forecast for the Next 48 Hours', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Predicted AQI')
plt.show()

# Plot the forecast components (trend, weekly, daily seasonality)
fig2 = model.plot_components(forecast, figsize=(15, 10))
plt.show()
```

-----

### \#\# Phase 4: Building an Interactive Dashboard with Streamlit

This final script takes our analysis and model and wraps it in a user-friendly web app.

**To run this:**

1.  Save the code below as a Python file (e.g., `app.py`).
2.  Make sure you have the merged data from Phase 2 saved as `ahmedabad_data.csv`.
3.  Run the command in your terminal: `streamlit run app.py`

<!-- end list -->

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

# --- App Configuration ---
st.set_page_config(
    page_title="Ahmedabad Urban Pulse Dashboard",
    page_icon="💨",
    layout="wide"
)

# --- Data Loading (Use the data generated from the previous step) ---
# In a real app, you'd load live data here.
@st.cache_data
def load_data():
    # We are re-using the data creation logic from the EDA script for this example
    # In a real project, you would save and load a CSV file
    end_date = pd.to_datetime('now', utc=True).tz_convert('Asia/Kolkata')
    start_date = end_date - pd.to_timedelta('30D')
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    data = {
        'timestamp': date_range,
        'aqi': np.random.randint(80, 250, size=len(date_range)),
        'travel_time_mins': np.random.randint(25, 75, size=len(date_range)),
        'temperature_c': np.random.uniform(28, 42, size=len(date_range)),
    }
    df = pd.DataFrame(data).set_index('timestamp')
    return df

df = load_data()
latest_data = df.iloc[-1]

# --- Model Training & Forecasting (Cached to run only once) ---
@st.cache_resource
def get_forecast(_df):
    prophet_df = _df.reset_index().rename(columns={'timestamp': 'ds', 'aqi': 'y'})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=48, freq='H')
    forecast = model.predict(future)
    return model, forecast

model, forecast = get_forecast(df)

# --- Dashboard UI ---
st.title("Ahmedabad's Urban Pulse 📈")
st.markdown("Live Air Quality and Traffic Analysis")

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current AQI", f"{int(latest_data['aqi'])}", "Satellite")
col2.metric("Current Travel Time", f"{int(latest_data['travel_time_mins'])} mins", "SG Highway -> Kalupur")
col3.metric("Current Temperature", f"{latest_data['temperature_c']:.1f} °C")

st.markdown("---")

# Charts
st.subheader("AQI Forecast for the Next 48 Hours")
fig_forecast = plot_plotly(model, forecast)
fig_forecast.update_layout(title="", xaxis_title="Date", yaxis_title="Predicted AQI")
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Historical Data (Last 30 Days)")
fig_historical = px.line(df, y="aqi", title="AQI Trend")
st.plotly_chart(fig_historical, use_container_width=True)

# Display Raw Data
if st.checkbox("Show Raw Data"):
    st.write(df.tail())
```