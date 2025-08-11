import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import snowflake.connector

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(page_title="📈 Financial Forecasting", layout="wide")
st.title("📈 Financial Forecasting App")
st.markdown("Using Prophet to forecast based on Snowflake data")

# -------------------
# Forecast slider
# -------------------
forecast_days = st.slider("📅 Select forecast horizon (days)", min_value=30, max_value=365, value=90, step=1)

# -------------------
# Snowflake Connection
# -------------------
try:
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
    )
    st.success("✅ Connected to Snowflake successfully!")
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# -------------------
# Fetch Data
# -------------------
query = "SELECT ds, y FROM forecast_data ORDER BY ds"
df = pd.read_sql(query, conn)
conn.close()

# -------------------
# Show Raw Data
# -------------------
st.subheader("📊 Raw Data")
st.write(df.tail())

# -------------------
# Data Preprocessing
# -------------------
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# -------------------
# Historical Trend Chart
# -------------------
st.subheader("📉 Historical Trend")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Actual Data"))
st.plotly_chart(fig_hist, use_container_width=True)

# -------------------
# Prophet Forecast
# -------------------
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# -------------------
# Forecast Chart
# -------------------
st.subheader("🔮 Forecasted Revenue")
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
st.plotly_chart(fig_forecast, use_container_width=True)

# -------------------
# Forecast Table
# -------------------
st.subheader("🧾 Forecast Table")
forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
forecast_table = forecast_table.rename(columns={
    "ds": "Date",
    "yhat": "Predicted Revenue",
    "yhat_lower": "Lower Bound",
    "yhat_upper": "Upper Bound"
})
st.dataframe(forecast_table)

# -------------------
# CSV Download
# -------------------
csv = forecast_table.to_csv(index=False)
st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")
