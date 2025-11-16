import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime as dt_datetime, timedelta as dt_timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 1. Configuration and Constants for NSE/Indian Market ---
APP_TITLE = "🇮🇳 Gold & Silver ETF Analysis (Indian Market)"

# Ticker symbols for popular Indian Metal ETFs on NSE (requires .NS suffix)
TICKER_GOLD = 'GOLDBEES.NS'    # Nippon India ETF Gold BeES
TICKER_SILVER = 'SILVER.NS'    # ICICI Prudential Silver ETF
PREDICTION_DAYS = 30           # Days to forecast
LOOKBACK_YEARS = 10            # Adjusted to 10 years for better ETF data availability

# --- 2. Data Fetching ---

@st.cache_data(ttl=3600)
def fetch_historical_data():
    """Fetches historical price data for Indian Gold and Silver ETFs."""
    
    # Generate dates robustly using standard datetime objects
    end_date = dt_datetime.now()
    start_date = end_date - dt_timedelta(days=LOOKBACK_YEARS * 365)
    
    # Format as strings only for the yf.download call
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    try:
        st.info("Fetching data for GOLDBEES.NS (Gold) and SILVER.NS (Silver) from NSE...")
        # Pass the formatted string dates
        gold_df = yf.download(TICKER_GOLD, start=start_date_str, end=end_date_str, progress=False)
        silver_df = yf.download(TICKER_SILVER, start=start_date_str, end=end_date_str, progress=False)
        
        data_df = pd.concat([
            gold_df['Close'].rename('Gold_ETF_Price'), 
            silver_df['Close'].rename('Silver_ETF_Price')
        ], axis=1).dropna()
        
        return data_df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}. Check ticker symbols or connection.")
        return pd.DataFrame()

# --- 3. Simple Linear Regression Model for Prediction ---

def predict_future_price(data_series, days_to_predict):
    """
    Predicts future price using a simple Linear Regression model, 
    based on the last year of data.
    """
    
    # Use the last year (approx. 252 trading days) of data for trend-based prediction
    recent_data = data_series.tail(252).to_frame(name='Price').copy() 
    
    # Create the Day feature (index for time)
    recent_data['Day'] = np.arange(len(recent_data))
    
    X = recent_data[['Day']]
    y = recent_data['Price']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Prepare future days for prediction
    last_day_index = recent_data['Day'].iloc[-1]
    future_days = np.array(range(last_day_index + 1, last_day_index + 1 + days_to_predict)).reshape(-1, 1)

    # Predict prices
    predicted_prices = model.predict(future_days)
    
    # Create future dates (using business day approximation 'B')
    last_date = recent_data.index[-1]
    future_dates = pd.date_range(start=last_date + dt_timedelta(days=1), periods=days_to_predict, freq='B')
    
    prediction_df = pd.DataFrame({
        'Predicted_Price': predicted_prices
    }, index=future_dates)
    
    return prediction_df

# --- 4. Streamlit UI (Main Function) ---

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Data source: Yahoo Finance (Indian Metal ETFs on NSE).")
    
    data_df = fetch_historical_data()

    if data_df.empty:
        st.stop() 

    # --- Sidebar Inputs ---
    st.sidebar.header("Prediction Settings")
    selected_metal = st.sidebar.selectbox(
        'Select Metal ETF for Forecast',
        ['Gold ETF (GOLDBEES.NS)', 'Silver ETF (SILVER.NS)']
    )
    days_to_forecast = st.sidebar.slider(
        'Days to Forecast',
        min_value=7,
        max_value=90,
        value=PREDICTION_DAYS
    )
    
    # Determine the series name based on selection
    if 'Gold' in selected_metal:
        series_name = 'Gold_ETF_Price'
        display_symbol = "GOLDBEES"
    else:
        series_name = 'Silver_ETF_Price'
        display_symbol = "SILVER.NS"

    # --- Live Rate Display ---
    current_date = data_df.index[-1].strftime('%Y-%m-%d')
    live_gold = data_df['Gold_ETF_Price'].iloc[-1].item()
    live_silver = data_df['Silver_ETF_Price'].iloc[-1].item()

    st.header("💰 Latest ETF Unit Price (₹)")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Date", current_date)
    col2.metric("Gold ETF (GOLDBEES)", f"₹{live_gold:,.2f}")
    col3.metric("Silver ETF (SILVER.NS)", f"₹{live_silver:,.2f}")
    
    st.divider()

    # --- Historical Plot ---
    st.header(f"🗓️ {LOOKBACK_YEARS}-Year Historical Trend")
    st.line_chart(data_df, use_container_width=True)

    # --- Prediction ---
    st.header(f"🔮 {selected_metal} Forecast ({days_to_forecast} Trading Days)")
    
    series = data_df[series_name]
        
    # Perform Prediction
    prediction_df = predict_future_price(series, days_to_forecast)
    
    # Combine actual and predicted for a nice chart
    plot_data = series.tail(100).to_frame(name='Actual Price')
    plot_data['Forecasted Price'] = np.nan
    
    # Connect the last actual point to the first predicted point
    last_actual_date = plot_data.index[-1]
    plot_data.loc[last_actual_date, 'Forecasted Price'] = series.loc[last_actual_date]
    
    combined_df = pd.concat([plot_data, prediction_df.rename(columns={'Predicted_Price': 'Forecasted Price'})[['Forecasted Price']]])
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    # Final Prediction Metrics
    current_rate = series.iloc[-1].item()
    final_forecast = prediction_df['Predicted_Price'].iloc[-1]
    delta = ((final_forecast - current_rate) / current_rate) * 100

    col_a, col_b = st.columns([1, 2])
    
    col_a.metric(
        f"Forecasted Price on {prediction_df.index[-1].strftime('%b %d, %Y')}",
        f"₹{final_forecast:,.2f}",
        f"{delta:+.2f}% change"
    )
    
    # Chart the forecast
    col_b.line_chart(combined_df, use_container_width=True)
    
    st.markdown("""
    **Disclaimer:** This forecast is based on historical price trend analysis using Linear Regression. It is **not** financial advice. 
    Metal ETF prices reflect the underlying commodity price in INR and are subject to market risks.
    """)

if __name__ == "__main__":
    main()
