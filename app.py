import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# Import Prophet
from prophet import Prophet
from prophet.plot import plot_plotly
# Import components with aliases for robust environment handling
from datetime import datetime as dt_datetime, timedelta as dt_timedelta
import warnings
warnings.filterwarnings("ignore") # Suppress common warnings

# --- 1. Configuration and Constants for BSE/NSE ---
APP_TITLE = "🧠 AI-Powered BSE/NSE Stock Price Predictor (Prophet)"

# Popular Indian Ticker Symbols (MUST use .NS for NSE, .BO for BSE)
TICKER_SYMBOLS = {
    'NSE': {
        'Nifty 50 (^NSEI)': '^NSEI',
        'Reliance Industries (RELIANCE.NS)': 'RELIANCE.NS',
        'TCS (TCS.NS)': 'TCS.NS',
        'HDFC Bank (HDFCBANK.NS)': 'HDFCBANK.NS',
    },
    'BSE': {
        'Sensex (^BSESN)': '^BSESN',
        'Reliance Industries (RELIANCE.BO)': 'RELIANCE.BO',
        'TCS (532540.BO)': '532540.BO',
        'HDFC Bank (500002.BO)': '500002.BO',
    }
}
PREDICTION_DAYS = 30          
LOOKBACK_YEARS = 5            # 5 years for stock analysis

# --- 2. Data Fetching ---

@st.cache_data(ttl=3600)
def fetch_historical_data(ticker_symbol):
    """Fetches historical price data and ensures the 'Price' column exists."""
    
    end_date_obj = dt_datetime.now()
    start_date_obj = end_date_obj - dt_timedelta(days=LOOKBACK_YEARS * 365)
    
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    end_date_str = end_date_obj.strftime('%Y-%m-%d')

    try:
        stock_df = yf.download(ticker_symbol, start=start_date_str, end=end_date_str, progress=False)
        
        if stock_df.empty:
             raise ValueError("Fetched data is empty.")
        
        # --- ROBUST COLUMN RENAMING ---
        if 'Close' in stock_df.columns:
            stock_df = stock_df.rename(columns={'Close': 'Price'})
        elif 'Adj Close' in stock_df.columns:
            stock_df = stock_df.rename(columns={'Adj Close': 'Price'})
        else:
            raise ValueError("Could not find a 'Close' or 'Adj Close' column.")

        stock_df = stock_df[['Price']]
        
        return stock_df
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}.")
        return pd.DataFrame()

# --- 3. Prophet AI Model for Forecasting ---

@st.cache_resource
def prophet_forecast(data_df, days_to_predict):
    """Trains Prophet model and forecasts future price."""
    
    # Prophet requires columns to be named 'ds' (datestamp) and 'y' (value)
    df_prophet = data_df.reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Price': 'y'})
    
    # Initialize and configure the Prophet model
    # Prophet can handle daily data and automatically fits trend/seasonality
    model = Prophet(
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        # Set a low change point prior scale for stable financial series
        changepoint_prior_scale=0.05 
    )

    # Fit the model
    model.fit(df_prophet)
    
    # Create a DataFrame for future dates
    future = model.make_future_dataframe(periods=days_to_predict, freq='B') # 'B' for business days

    # Make prediction
    forecast = model.predict(future)
    
    # Extract the last predicted values for the forecast window
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict).copy()
    
    # Rename columns and set index
    forecast_df = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Price'})
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df, model

# --- 4. Streamlit UI (Main Function) ---

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("AI Model: **Prophet** (Meta). Prediction utilizes 5 years of historical data to capture trend and seasonality.")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("Stock Selection & Settings")
    
    selected_exchange = st.sidebar.radio('Select Exchange', ('NSE', 'BSE'))
    
    ticker_name = st.sidebar.selectbox(
        f'Select Stock/Index ({selected_exchange})',
        list(TICKER_SYMBOLS[selected_exchange].keys())
    )
    ticker_symbol = TICKER_SYMBOLS[selected_exchange][ticker_name]

    days_to_forecast = st.sidebar.slider(
        'Days to Forecast',
        min_value=7,
        max_value=90,
        value=PREDICTION_DAYS
    )
    
    # Fetch Data
    with st.spinner(f"Fetching {ticker_symbol} data..."):
        data_df = fetch_historical_data(ticker_symbol)

    if data_df.empty:
        st.stop() 

    # --- Run AI Prediction ---
    with st.spinner("Training Prophet AI model and generating forecast..."):
        forecast_df, model = prophet_forecast(data_df, days_to_forecast)
    
    # --- Live Rate Display ---
    current_date = data_df.index[-1].strftime('%Y-%m-%d')
    live_price = data_df['Price'].iloc[-1].item()
    
    if len(data_df) >= 2:
        previous_price = data_df['Price'].iloc[-2].item()
        change = live_price - previous_price
        change_percent = (change / previous_price) * 100
    else:
        change = 0
        change_percent = 0
        
    st.header(f"💰 {ticker_name} Live Price (₹)")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Date", current_date)
    col2.metric("Latest Price", f"₹{live_price:,.2f}", delta=f"₹{change:+.2f}")
    col3.metric("Daily Change (%)", f"{change_percent:+.2f}%")
    
    st.divider()

    # --- Prediction Summary ---
    st.header(f"🔮 AI Forecast for {ticker_name} ({days_to_forecast} Trading Days)")
    
    final_forecast = forecast_df['Predicted_Price'].iloc[-1]
    prediction_delta = ((final_forecast - live_price) / live_price) * 100
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.metric(
            f"Forecasted Price on {forecast_df.index[-1].strftime('%b %d, %Y')}",
            f"₹{final_forecast:,.2f}",
            f"{prediction_delta:+.2f}% change",
            delta_color="off" 
        )
        st.dataframe(forecast_df[['Predicted_Price']].style.format({"Predicted_Price": "₹{:,.2f}"}), use_container_width=True)

    with col_b:
        # Plot the forecast using Prophet's built-in plotly function
        fig1 = plot_plotly(model, forecast_df.reset_index())
        fig1.update_layout(
            title=f"Prophet Forecast for {ticker_name}",
            xaxis_title="Date",
            yaxis_title="Closing Price (₹)",
            showlegend=False,
            height=450,
            # Add trace of historical data for context
            shapes=[dict(
                type="line",
                xref="x", yref="y",
                x0=data_df.index[-1], y0=data_df['Price'].iloc[-1],
                x1=data_df.index[-1], y1=data_df['Price'].iloc[-1],
                line=dict(color="red", width=2, dash="dash")
            )]
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    **Disclaimer:** This forecast is generated by the **Prophet AI model**, which excels at capturing trend and seasonality. 
    It is a simplified model for demonstration purposes only and should **NOT** be used for actual investment or trading decisions.
    """)

if __name__ == "__main__":
    main()
