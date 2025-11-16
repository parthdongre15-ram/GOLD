import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 1. Configuration and Constants ---
APP_TITLE = "📈 Gold & Silver Price Analysis (20-Year History & Forecast)"
TICKER_GOLD = 'GC=F'    # Gold Futures (USD/Ounce)
TICKER_SILVER = 'SI=F'  # Silver Futures (USD/Ounce)
PREDICTION_DAYS = 30    # Days to forecast
LOOKBACK_YEARS = 20

# --- 2. Data Fetching ---

@st.cache_data(ttl=3600)
def fetch_historical_data():
    """Fetches historical price data for Gold and Silver."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)).strftime('%Y-%m-%d')
    
    try:
        gold_df = yf.download(TICKER_GOLD, start=start_date, end=end_date, progress=False)
        silver_df = yf.download(TICKER_SILVER, start=start_date, end=end_date, progress=False)
        
        data_df = pd.concat([
            gold_df['Close'].rename('Gold_Price'), 
            silver_df['Close'].rename('Silver_Price')
        ], axis=1).dropna()
        
        return data_df
    except Exception as e:
        st.error(f"Error fetching data: {e}. Check ticker symbols or connection.")
        return pd.DataFrame()

# --- 3. Simple Linear Regression Model for Prediction ---

def predict_future_price(data_series, days_to_predict):
    """
    Predicts future price using a simple Linear Regression model, 
    based on the last 365 days of data.
    """
    
    # Use the last year of data for trend-based prediction
    recent_data = data_series.tail(252).to_frame(name='Price')
    
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
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict, freq='B')
    
    prediction_df = pd.DataFrame({
        'Predicted_Price': predicted_prices
    }, index=future_dates)
    
    return prediction_df

# --- 4. Streamlit UI (Main Function) ---

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Data source: Yahoo Finance (20-year history of Gold and Silver Futures - USD/Ounce). Prediction uses Linear Regression on last year's data.")
    
    data_df = fetch_historical_data()

    if data_df.empty:
        return

    # --- Sidebar Inputs ---
    st.sidebar.header("Prediction Settings")
    selected_metal = st.sidebar.selectbox(
        'Select Metal for Forecast',
        ['Gold', 'Silver']
    )
    days_to_forecast = st.sidebar.slider(
        'Days to Forecast',
        min_value=7,
        max_value=90,
        value=PREDICTION_DAYS
    )
    
    # --- Live Rate Display ---
    current_date = data_df.index[-1].strftime('%Y-%m-%d')
    live_gold = data_df['Gold_Price'].iloc[-1].item()
    live_silver = data_df['Silver_Price'].iloc[-1].item()

    st.header("💰 Live Rates & Key Data")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Date", current_date)
    col2.metric("Gold Price (USD/Oz)", f"${live_gold:,.2f}")
    col3.metric("Silver Price (USD/Oz)", f"${live_silver:,.2f}")
    
    st.divider()

    # --- Historical Plot ---
    st.header("🗓️ 20-Year Historical Trend")
    st.line_chart(data_df, use_container_width=True)

    # --- Prediction ---
    st.header(f"🔮 {selected_metal} Forecast ({days_to_forecast} Days)")

    if selected_metal == 'Gold':
        series = data_df['Gold_Price']
    else:
        series = data_df['Silver_Price']
        
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
        f"Forecast on {prediction_df.index[-1].strftime('%b %d, %Y')}",
        f"${final_forecast:,.2f}",
        f"{delta:+.2f}% change"
    )
    
    # Chart the forecast
    col_b.line_chart(combined_df, use_container_width=True)
    
    st.markdown("""
    **Disclaimer:** This is a simple trend prediction for academic purposes only. 
    It is **not** financial advice. Gold and silver prices are highly volatile and influenced by complex global factors (interest rates, inflation, geopolitics).
    """)

if __name__ == "__main__":
    main()
