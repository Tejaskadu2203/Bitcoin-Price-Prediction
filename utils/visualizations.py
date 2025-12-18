import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple
import datetime

def plot_bitcoin_history(data: pd.DataFrame, show_volume: bool = True, show_ma: bool = True) -> go.Figure:
    """
    Create an interactive plot of Bitcoin price history.
    
    Args:
        data: DataFrame with historical Bitcoin data
        show_volume: Whether to display volume data
        show_ma: Whether to display moving averages
        
    Returns:
        Plotly figure object with the historical chart
    """
    # Create subplots
    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Add price trace
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['price'],
        mode='lines',
        name='Bitcoin Price',
        line=dict(color='#F7931A', width=2)
    ), row=1, col=1)
    
    # Add moving averages if requested
    if show_ma:
        ma_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ma_periods = [7, 14, 30, 90]
        
        for i, period in enumerate(ma_periods):
            # Calculate moving average
            if len(data) >= period:
                ma_name = f'SMA{period}'
                if ma_name not in data.columns:
                    data[ma_name] = data['price'].rolling(window=period).mean()
                
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data[ma_name],
                    mode='lines',
                    name=f'{period}-day MA',
                    line=dict(color=ma_colors[i % len(ma_colors)], width=1, dash='dot'),
                    opacity=0.7
                ), row=1, col=1)
    
    # Add volume if requested
    if show_volume and 'volume' in data.columns:
        fig.add_trace(go.Bar(
            x=data['timestamp'],
            y=data['volume'],
            name='Volume',
            marker=dict(color='rgba(200, 200, 200, 0.5)')
        ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Price History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis_tickprefix='$',
        legend=dict(orientation='h', y=1.1),
        height=700,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
    
    # Add buttons for time ranges
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(step='all', label='All')
                ])
            )
        )
    )
    
    return fig

def plot_predictions(data: pd.DataFrame, future_dates: List, predicted_prices: List) -> go.Figure:
    """
    Create a plot of historical prices and predictions.
    
    Args:
        data: DataFrame with historical data
        future_dates: List of future dates for predictions
        predicted_prices: List of predicted prices
        
    Returns:
        Plotly figure with historical data and predictions
    """
    # Use the most recent data for display (last 60 days)
    recent_days = min(60, len(data))
    recent_data = data.iloc[-recent_days:]
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add historical price line
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data['price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#F7931A', width=3, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Calculate confidence intervals
    std_dev = recent_data['price'].std() * 0.1
    upper_bound = [price + std_dev * (i+1) for i, price in enumerate(predicted_prices)]
    lower_bound = [price - std_dev * (i+1) for i, price in enumerate(predicted_prices)]
    
    # Add confidence interval as a filled area
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        marker=dict(color="#F7931A"),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        marker=dict(color="#F7931A"),
        fillcolor='rgba(247, 147, 26, 0.2)',
        fill='tonexty',
        showlegend=False
    ))
    
    # Add connecting line between last historical price and first prediction
    fig.add_trace(go.Scatter(
        x=[recent_data['timestamp'].iloc[-1], future_dates[0]],
        y=[recent_data['price'].iloc[-1], predicted_prices[0]],
        mode='lines',
        line=dict(color='#F7931A', width=1.5, dash='dot'),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis_tickprefix='$',
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    # Add vertical line marking prediction start
    prediction_start = future_dates[0]
    fig.add_vline(
        x=prediction_start,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )
    
    # Add annotation for prediction start
    fig.add_annotation(
        x=prediction_start,
        y=predicted_prices[0] * 1.1,
        text="Prediction Start",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="gray"
    )
    
    return fig

def plot_accuracy_metrics(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create a visual representation of prediction accuracy metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        
    Returns:
        Plotly figure with accuracy metrics visualization
    """
    # Create subplot grid for multiple metrics
    fig = make_subplots(
        rows=2, 
        cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Combined Accuracy", "Directional Accuracy", 
                        "Price Accuracy", "Error Metrics")
    )
    
    # Add combined accuracy gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['combined_accuracy'],
        title={'text': "Combined Accuracy"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': 'red'},
                {'range': [60, 80], 'color': 'orange'},
                {'range': [80, 90], 'color': 'yellow'},
                {'range': [90, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ), row=1, col=1)
    
    # Add directional accuracy gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['directional_accuracy'],
        title={'text': "Directional Accuracy"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': 'red'},
                {'range': [60, 80], 'color': 'orange'},
                {'range': [80, 90], 'color': 'yellow'},
                {'range': [90, 100], 'color': 'green'}
            ]
        }
    ), row=1, col=2)
    
    # Add price accuracy gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['price_accuracy'],
        title={'text': "Price Accuracy"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': 'red'},
                {'range': [60, 80], 'color': 'orange'},
                {'range': [80, 90], 'color': 'yellow'},
                {'range': [90, 100], 'color': 'green'}
            ]
        }
    ), row=2, col=1)
    
    # Add MAE and RMSE metric
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=metrics['mae'],
        title={'text': "MAE / RMSE"},
        delta={'reference': metrics['rmse'], 'relative': True},
        number={'prefix': "$"}
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=600,
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        template='plotly_white'
    )
    
    return fig
