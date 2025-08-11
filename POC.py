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

# === NEW: Best fit forecast chart with shaded confidence interval === #
st.subheader("🔮 Forecasted Revenue (Best Fit Line)")

# Separate historical and forecast parts
historical = forecast[forecast['ds'] <= df['ds'].max()]
future_forecast = forecast[forecast['ds'] > df['ds'].max()]

fig = go.Figure()

# Historical line
fig.add_trace(go.Scatter(
    x=historical['ds'], y=historical['yhat'],
    mode='lines',
    name='Historical',
    line=dict(color='blue', width=2)
))

# Forecast line
fig.add_trace(go.Scatter(
    x=future_forecast['ds'], y=future_forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='red', width=3, dash='dash')
))

# Confidence interval shading
fig.add_trace(go.Scatter(
    x=list(future_forecast['ds']) + list(future_forecast['ds'])[::-1],
    y=list(future_forecast['yhat_upper']) + list(future_forecast['yhat_lower'])[::-1],
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Confidence Interval'
))

fig.update_layout(
    title="Forecasted Revenue with Confidence Interval",
    xaxis_title="Date",
    yaxis_title="Revenue",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# === Forecast Table === #
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

# === Export as CSV === #
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_csv(index=False)
st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")
