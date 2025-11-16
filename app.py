import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# Import components with aliases for robust environment handling
from datetime import datetime as dt_datetime, timedelta as dt_timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore") # Ignore common pandas/sklearn warnings

# --- 1. Configuration and Constants for BSE/NSE ---
APP_TITLE = "🇮🇳 BSE & NSE Stock Price Predictor"

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
PREDICTION_DAYS = 7           
LOOKBACK_YEARS = 5            # 5 years for stock analysis

# --- 2. Data Fetching ---

@st.cache_data(ttl=3600)
def fetch_historical_data(ticker_symbol):
    """
    Fetches historical price data and ensures the 'Price' column exists
    to prevent the KeyError during plotting.
    """
    
    # Create datetime objects
    end_date_obj = dt_datetime.now()
    start_date_obj = end_date_obj - dt_timedelta(days=LOOKBACK_YEARS * 365)
    
    # Format as strings explicitly for yfinance
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    end_date_str = end_date_obj.strftime('%Y-%m-%d')

    try:
        st.info(f"Fetching data for {ticker_symbol} from {start_date_str} to {end_date_str}...")
        
        stock_df = yf.download(ticker_symbol, start=start_date_str, end=end_date_str, progress=False)
        
        if stock_df.empty:
             raise ValueError("Fetched data is empty. Check ticker symbol or date range.")
        
        # --- FIX: ROBUST COLUMN RENAMING to prevent KeyError 'Price' ---
        if 'Close' in stock_df.columns:
            # Most common column name for closing price
            stock_df = stock_df.rename(columns={'Close': 'Price'})
        elif 'Adj Close' in stock_df.columns:
            # Fallback for adjusted closing price
            stock_df = stock_df.rename(columns={'Adj Close': 'Price'})
        else:
            # If neither is found, raise an error
            raise ValueError("Could not find a 'Close' or 'Adj Close' column in the data.")

        # Explicitly reduce the DataFrame to only the guaranteed 'Price' column
        stock_df = stock_df[['Price']]
        
        return stock_df
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}. Please check the ticker and the Yahoo Finance data availability.")
        return pd.DataFrame()

# --- 3. Simple Linear Regression Model for Prediction ---

def predict_future_price(data_series, days_to_predict):
    """
    Predicts future price using a simple Linear Regression model, 
    based on the last year (approx. 252 trading days) of data.
    """
    
    # Use the last year of data for trend-based prediction
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
    st.caption("Data source: Yahoo Finance (BSE/NSE). Prediction uses Linear Regression on 5 years of historical data.")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("Stock Selection & Settings")
    
    # Exchange Selection
    selected_exchange = st.sidebar.radio(
        'Select Exchange',
        ('NSE', 'BSE')
    )
    
    # Ticker Selection based on Exchange
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

    # --- Live Rate Display ---
    current_date = data_df.index[-1].strftime('%Y-%m-%d')
    live_price = data_df['Price'].iloc[-1].item()
    
    # Calculate 1-day change
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

    # --- Historical Plot ---
    st.header(f"🗓️ {LOOKBACK_YEARS}-Year Historical Price Trend")
    # Pass the DataFrame directly since it only contains the 'Price' column
    st.line_chart(data_df, use_container_width=True)

    # --- Prediction ---
    st.header(f"🔮 Price Forecast for {ticker_name} ({days_to_forecast} Trading Days)")
    
    series = data_df['Price']
        
    # Perform Prediction
    prediction_df = predict_future_price(series, days_to_forecast)
    
    # Combine actual and predicted for a nice chart
    plot_data = series.tail(90).to_frame(name='Actual Price')
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
    **Disclaimer:** This forecast uses a simple Linear Regression model and is for academic/demonstration purposes only. 
    It is **not** financial advice. Stock prices are highly volatile and complex.
    """)

if __name__ == "__main__":
    main()
