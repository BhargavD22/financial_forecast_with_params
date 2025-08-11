# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from prophet import Prophet
import snowflake.connector
import plotly.graph_objects as go

# === Function to connect and fetch data from Snowflake === #
@st.cache_data
def get_snowflake_data():
    conn = snowflake.connector.connect(
        user=st.secrets["USER"],
        password=st.secrets["PASSWORD"],
        account=st.secrets["ACCOUNT"],
        warehouse=st.secrets["WAREHOUSE"],
        database=st.secrets["DATABASE"],
        schema=st.secrets["SCHEMA"]
    )
    query = """
        SELECT ds AS "ds", y AS "y"
        FROM forecast_data
        WHERE ds >= '2020-01-01'
        ORDER BY ds;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# === Streamlit App UI === #
st.title("📈 Financial Forecasting App (Snowflake + Prophet)")
st.markdown("This app retrieves financial data from Snowflake and forecasts future revenue using Prophet.")

# Forecast input
forecast_days = st.slider("Select number of days to forecast:", min_value=30, max_value=365, value=90)

# Load data
with st.spinner("Connecting to Snowflake and fetching data..."):
    df = get_snowflake_data()

# Check and preview data
st.subheader("📊 Historical Data")
df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format
st.line_chart(df.set_index('ds')['y'])

# Fit Prophet model
model = Prophet()
model.fit(df)

# Make forecast
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# Show forecast chart
st.subheader("🔮 Forecasted Revenue")
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
st.plotly_chart(fig, use_container_width=True)

# Forecast Table
st.subheader("🧾 Forecast Table")
st.dataframe(
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).rename(
        columns={
            "ds": "Date",
            "yhat": "Predicted Revenue",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound"
        }
    )
)

# Export as CSV
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_csv(index=False)
st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")
