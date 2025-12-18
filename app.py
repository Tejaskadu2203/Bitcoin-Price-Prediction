from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import time

# Import custom modules
from utils.data_fetcher import fetch_bitcoin_data, get_current_price, prepare_features
from utils.prediction import BitcoinPricePredictor, optimize_predictor, combine_predictions
from utils.visualizations_flask import plot_bitcoin_history, prepare_prediction_data, prepare_accuracy_metrics

app = Flask(__name__)

# Cache for data to avoid repeated API calls
cache = {
    'bitcoin_data': None,
    'last_fetch': 0,
    'current_price': 0,
    'current_price_time': 0,
    'predictions': None,
    'prediction_time': 0
}

def load_bitcoin_data(days=365, force_refresh=False):
    """Load Bitcoin price data and cache it"""
    current_time = time.time()
    # Cache for 1 hour (3600 seconds)
    if not force_refresh and cache['bitcoin_data'] is not None and current_time - cache['last_fetch'] < 3600:
        return cache['bitcoin_data']
    
    data = fetch_bitcoin_data(days=days)
    if not data.empty:
        cache['bitcoin_data'] = data
        cache['last_fetch'] = current_time
    return data

def get_latest_price(force_refresh=False):
    """Get the latest Bitcoin price"""
    current_time = time.time()
    # Cache for 5 minutes (300 seconds)
    if not force_refresh and cache['current_price'] > 0 and current_time - cache['current_price_time'] < 300:
        return cache['current_price']
    
    price = get_current_price()
    if price > 0:
        cache['current_price'] = price
        cache['current_price_time'] = current_time
    return price

@app.route('/')
def index():
    """Render the main page"""
    # Load sample data for initial display
    current_price = get_latest_price()
    return render_template('index.html', current_price=current_price)

@app.route('/api/historical-data')
def historical_data():
    """API endpoint to get historical Bitcoin data"""
    days = request.args.get('days', default=90, type=int)
    show_volume = request.args.get('show_volume', default='true', type=str).lower() == 'true'
    show_ma = request.args.get('show_ma', default='true', type=str).lower() == 'true'
    
    data = load_bitcoin_data(days=days)
    
    if data.empty:
        return jsonify({'error': 'Failed to load data'}), 500
    
    # Use our visualization function to format the data
    result = plot_bitcoin_history(data, show_volume=show_volume, show_ma=show_ma)
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to generate predictions"""
    # Get parameters from request
    data = request.get_json()
    hist_range = data.get('histRange', 180)
    pred_days = data.get('predDays', 3)
    target_accuracy = data.get('targetAccuracy', 95.0)
    use_ensemble = data.get('useEnsemble', True)
    
    # Model weights
    if use_ensemble:
        ml_weight = data.get('mlWeight', 0.6)
        arima_weight = data.get('arimaWeight', 0.2)
        es_weight = data.get('esWeight', 0.2)
        
        # Normalize weights
        total = ml_weight + arima_weight + es_weight
        if total == 0:
            ml_weight, arima_weight, es_weight = 0.6, 0.2, 0.2
        else:
            ml_weight /= total
            arima_weight /= total
            es_weight /= total
            
        model_weights = {
            'ml_ensemble': ml_weight,
            'arima': arima_weight,
            'exp_smoothing': es_weight
        }
    else:
        model_weights = {'ml_ensemble': 1.0}
    
    # Step 1: Load data
    bitcoin_data = load_bitcoin_data(days=hist_range)
    
    if bitcoin_data.empty:
        return jsonify({'error': 'Failed to load Bitcoin data'}), 500
    
    # Step 2: Prepare features
    featured_data = prepare_features(bitcoin_data)
    
    # Step 3: Optimize and train the model
    if use_ensemble:
        future_dates, predicted_prices, pred_df = combine_predictions(
            featured_data, 
            days_to_predict=pred_days,
            weights=model_weights
        )
        
        # For accuracy metrics, we'll use the primary model
        predictor = BitcoinPricePredictor()
        predictor.train(featured_data, days_ahead=1)
        accuracy_metrics = predictor.accuracy_metrics
    else:
        # Use a single optimized model
        predictor = optimize_predictor(
            featured_data, 
            days_ahead=pred_days,
            target_accuracy=target_accuracy
        )
        
        # Generate predictions
        pred_df = predictor.predict(days=pred_days)
        future_dates = pred_df['date'].tolist()
        predicted_prices = pred_df['predicted_price'].tolist()
        accuracy_metrics = predictor.accuracy_metrics
    
    # Get current price
    current_price = get_latest_price()
    
    # Format dates for JSON
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
    
    # Create detailed prediction table
    prediction_table = []
    for i, (date, price) in enumerate(zip(future_dates, predicted_prices)):
        day_label = f"Day {i+1}" if i > 0 else "Tomorrow"
        formatted_date = date.strftime("%Y-%m-%d")
        change_pct = (price / (predicted_prices[i-1] if i > 0 else current_price) - 1) * 100
        prediction_table.append({
            "day": day_label,
            "date": formatted_date,
            "predictedPrice": price,
            "change": change_pct
        })
    
    # Store the results in a dictionary
    result = {
        'current_price': current_price,
        'price_change': (current_price / bitcoin_data['price'].iloc[-1] - 1) * 100 if not bitcoin_data.empty else 0,
        'last_prediction': predicted_prices[-1],
        'prediction_change': (predicted_prices[-1] / current_price - 1) * 100,
        'future_dates': future_dates_str,
        'predicted_prices': predicted_prices,
        'prediction_table': prediction_table,
        'accuracy_metrics': accuracy_metrics,
        'model_weights': model_weights
    }
    
    # Update cache
    cache['predictions'] = result
    cache['prediction_time'] = time.time()
    
    return jsonify(result)

@app.route('/api/current-price')
def current_price():
    """API endpoint to get current Bitcoin price"""
    price = get_latest_price()
    return jsonify({'price': price})

@app.route('/api/sample-data')
def sample_data():
    """API endpoint to get sample data for initial display"""
    data = load_bitcoin_data(days=90)
    if data.empty:
        return jsonify({'error': 'Failed to load data'}), 500
    
    # Use our visualization function
    result = plot_bitcoin_history(data, show_volume=True, show_ma=True)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
