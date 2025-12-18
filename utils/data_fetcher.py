import requests
import pandas as pd
import numpy as np
import time
import datetime
from typing import Dict, Any, Optional

def make_api_request(url, params=None, max_retries=3, base_delay=2):
    """
    Makes an API request with exponential backoff for retries to handle rate limiting.
    
    Args:
        url: API endpoint URL
        params: Query parameters for the request
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
    
    Returns:
        JSON response if successful
    
    Raises:
        requests.exceptions.RequestException: If request fails after all retries
    """
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Bitcoin-Price-Predictor/1.0'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            # Calculate exponential backoff delay
            delay = base_delay * (2 ** attempt)
            print(f"API request failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    # This should never be reached due to the exception in the loop,
    # but adding as a safeguard
    raise requests.exceptions.RequestException("All retry attempts failed")

def fetch_bitcoin_data(days=365):
    """
    Fetch historical Bitcoin price data from CoinGecko API.
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with historical Bitcoin data
    """
    # Calculate dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Convert dates to Unix timestamps (seconds)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    # CoinGecko API endpoint for Bitcoin market chart
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    params = {
        'vs_currency': 'usd',
        'from': start_timestamp,
        'to': end_timestamp
    }
    
    try:
        data = make_api_request(url, params)
        
        # Extract price and volume data
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            return pd.DataFrame()
        
        # Create DataFrames for prices and volumes
        price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        # Convert Unix timestamps to datetime
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        
        # Merge price and volume DataFrames
        df = pd.merge_asof(price_df, volume_df, on='timestamp')
        
        # Resample to daily data (end of day)
        df = df.set_index('timestamp')
        daily_df = df.resample('D').agg({
            'price': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Reset index to make timestamp a column again
        daily_df = daily_df.reset_index()
        
        return daily_df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bitcoin data: {e}")
        return pd.DataFrame()

def get_current_price():
    """
    Get the current Bitcoin price from CoinGecko API.
    
    Returns:
        Current Bitcoin price in USD
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': 'bitcoin',
        'vs_currencies': 'usd'
    }
    
    try:
        data = make_api_request(url, params)
        return data.get('bitcoin', {}).get('usd', 0)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current Bitcoin price: {e}")
        return 0

def get_bitcoin_ohlc(days=7):
    """
    Get Bitcoin OHLC (Open, High, Low, Close) data for the specified number of days.
    
    Args:
        days: Number of days of data to fetch (1, 7, 14, 30, 90, 180, 365)
        
    Returns:
        DataFrame with OHLC data
    """
    # Valid values for the days parameter
    valid_days = [1, 7, 14, 30, 90, 180, 365]
    if days not in valid_days:
        days = min(valid_days, key=lambda x: abs(x - days))
    
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    
    try:
        data = make_api_request(url, params)
        
        # Create DataFrame
        ohlc_df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        
        # Convert Unix timestamps to datetime
        ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp'], unit='ms')
        
        return ohlc_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bitcoin OHLC data: {e}")
        return pd.DataFrame()

def prepare_features(df):
    """
    Create features for the prediction models
    
    Args:
        df: DataFrame with historical Bitcoin data
    
    Returns:
        DataFrame with additional features for modeling
    """
    data = df.copy()
    
    # Basic price features
    data['price_yesterday'] = data['price'].shift(1)
    data['price_2days_ago'] = data['price'].shift(2)
    data['price_3days_ago'] = data['price'].shift(3)
    data['price_4days_ago'] = data['price'].shift(4)
    data['price_5days_ago'] = data['price'].shift(5)
    data['price_7days_ago'] = data['price'].shift(7)
    
    # Price changes (momentum)
    data['price_change_1d'] = data['price'] / data['price_yesterday'] - 1
    data['price_change_2d'] = data['price'] / data['price_2days_ago'] - 1
    data['price_change_3d'] = data['price'] / data['price_3days_ago'] - 1
    data['price_change_5d'] = data['price'] / data['price_5days_ago'] - 1
    data['price_change_7d'] = data['price'] / data['price_7days_ago'] - 1
    
    # Simple moving averages
    data['SMA5'] = data['price'].rolling(window=5).mean()
    data['SMA7'] = data['price'].rolling(window=7).mean()
    data['SMA14'] = data['price'].rolling(window=14).mean()
    data['SMA30'] = data['price'].rolling(window=30).mean()
    data['SMA60'] = data['price'].rolling(window=60).mean()
    
    # Exponential moving averages
    data['EMA5'] = data['price'].ewm(span=5, adjust=False).mean()
    data['EMA12'] = data['price'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['price'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Relative strength between moving averages
    data['SMA_ratio_5_30'] = data['SMA5'] / data['SMA30']
    data['SMA_ratio_7_30'] = data['SMA7'] / data['SMA30']
    
    # Volatility features
    data['volatility_7d'] = data['price'].rolling(window=7).std()
    data['volatility_14d'] = data['price'].rolling(window=14).std()
    data['volatility_ratio'] = data['volatility_7d'] / data['volatility_14d']
    
    # Volume features 
    data['volume_SMA5'] = data['volume'].rolling(window=5).mean()
    data['volume_SMA10'] = data['volume'].rolling(window=10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_SMA5']
    data['volume_change'] = data['volume'].pct_change()
    
    # Rate of change
    data['ROC_5'] = (data['price'] / data['price'].shift(5) - 1) * 100
    data['ROC_10'] = (data['price'] / data['price'].shift(10) - 1) * 100
    
    # Drop rows with NaN values created by the shifting/rolling operations
    data = data.dropna()
    
    return data
