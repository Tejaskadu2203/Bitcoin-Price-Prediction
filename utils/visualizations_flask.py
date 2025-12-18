import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json

def plot_bitcoin_history(data: pd.DataFrame, show_volume: bool = True, show_ma: bool = True) -> Dict:
    """
    Prepare data for plotting historical Bitcoin prices with JavaScript/Plotly.
    
    Args:
        data: DataFrame with historical Bitcoin data
        show_volume: Whether to display volume data
        show_ma: Whether to display moving averages
        
    Returns:
        Dictionary with data for plotting
    """
    # Extract basic data
    plot_data = {
        'dates': data['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        'prices': data['price'].tolist(),
        'volumes': data['volume'].tolist() if 'volume' in data.columns else [],
        'show_volume': show_volume,
        'show_ma': show_ma
    }
    
    # Add moving averages if available
    if show_ma:
        ma_columns = [col for col in data.columns if col.startswith('SMA')]
        for col in ma_columns:
            plot_data[col] = data[col].tolist()
    
    return plot_data

def prepare_prediction_data(data: pd.DataFrame, future_dates: List, predicted_prices: List) -> Dict:
    """
    Prepare data for plotting predictions with JavaScript/Plotly.
    
    Args:
        data: DataFrame with historical data
        future_dates: List of future dates for predictions
        predicted_prices: List of predicted prices
        
    Returns:
        Dictionary with data for plotting
    """
    # Extract the most recent historical data (last 60 days)
    recent_days = min(60, len(data))
    recent_data = data.iloc[-recent_days:]
    
    # Format dates for JSON
    historical_dates = recent_data['timestamp'].dt.strftime('%Y-%m-%d').tolist()
    historical_prices = recent_data['price'].tolist()
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
    
    # Calculate confidence intervals
    std_dev = data['price'].std() * 0.1
    upper_bound = [price + std_dev * (i+1) for i, price in enumerate(predicted_prices)]
    lower_bound = [price - std_dev * (i+1) for i, price in enumerate(predicted_prices)]
    
    # Prepare plot data
    plot_data = {
        'historical_dates': historical_dates,
        'historical_prices': historical_prices,
        'future_dates': future_dates_str,
        'predicted_prices': predicted_prices,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'last_historical_date': data['timestamp'].iloc[-1].strftime('%Y-%m-%d'),
        'last_historical_price': data['price'].iloc[-1]
    }
    
    return plot_data

def prepare_accuracy_metrics(metrics: Dict[str, Any]) -> Dict:
    """
    Prepare accuracy metrics for JavaScript/Plotly visualization.
    
    Args:
        metrics: Dictionary with evaluation metrics
        
    Returns:
        Dictionary with formatted metrics for display
    """
    # Extract and format metrics
    formatted_metrics = {
        'combined_accuracy': metrics.get('combined_accuracy', 0),
        'directional_accuracy': metrics.get('directional_accuracy', 0),
        'price_accuracy': metrics.get('price_accuracy', 0),
        'mae': metrics.get('mae', 0),
        'rmse': metrics.get('rmse', 0)
    }
    
    return formatted_metrics
