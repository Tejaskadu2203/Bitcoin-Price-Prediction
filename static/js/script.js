document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI elements
    initializeTabs();
    initializeFormListeners();
    updateRangeLabels();
    toggleWeightSettings();
    loadInitialData();
    
    // Set up dark mode toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    darkModeToggle.addEventListener('click', toggleDarkMode);
    
    // Check if dark mode is enabled in local storage
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
    }
    
    // Set up form submission
    const predictionForm = document.getElementById('prediction-form');
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        generatePrediction();
    });
    
    // Set up historical chart controls
    const showVolume = document.getElementById('show-volume');
    const showMA = document.getElementById('show-ma');
    
    showVolume.addEventListener('change', function() {
        updateHistoricalChart();
    });
    
    showMA.addEventListener('change', function() {
        updateHistoricalChart();
    });
    
    // Poll for current price updates
    setInterval(updateCurrentPrice, 60000); // Update every minute
});

function initializeTabs() {
    const triggerTabList = document.querySelectorAll('#mainTabs button');
    triggerTabList.forEach(triggerEl => {
        triggerEl.addEventListener('click', function(event) {
            event.preventDefault();
            const tabTrigger = new bootstrap.Tab(triggerEl);
            tabTrigger.show();
        });
    });
}

function initializeFormListeners() {
    // Add event listeners to range inputs
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        input.addEventListener('input', updateRangeLabels);
    });
    
    // Add listener to ensemble checkbox
    const useEnsemble = document.getElementById('useEnsemble');
    useEnsemble.addEventListener('change', toggleWeightSettings);
}

function updateRangeLabels() {
    // Update labels for all range inputs
    document.getElementById('histRangeValue').textContent = document.getElementById('histRange').value;
    document.getElementById('predDaysValue').textContent = document.getElementById('predDays').value;
    document.getElementById('targetAccuracyValue').textContent = document.getElementById('targetAccuracy').value + '%';
    document.getElementById('mlWeightValue').textContent = document.getElementById('mlWeight').value;
    document.getElementById('arimaWeightValue').textContent = document.getElementById('arimaWeight').value;
    document.getElementById('esWeightValue').textContent = document.getElementById('esWeight').value;
}

function toggleWeightSettings() {
    const useEnsemble = document.getElementById('useEnsemble').checked;
    const weightSettings = document.getElementById('weightSettings');
    
    if (useEnsemble) {
        weightSettings.classList.remove('d-none');
    } else {
        weightSettings.classList.add('d-none');
    }
}

function loadInitialData() {
    // Fetch sample data for historical chart
    fetch('/api/sample-data')
        .then(response => response.json())
        .then(data => {
            if (data && data.dates) {
                plotHistoricalChart(data);
                populateHistoricalTable(data);
            }
        })
        .catch(error => {
            console.error('Error loading sample data:', error);
        });
    
    // Get current price
    updateCurrentPrice();
}

function updateCurrentPrice() {
    fetch('/api/current-price')
        .then(response => response.json())
        .then(data => {
            if (data && data.price) {
                const priceElements = document.querySelectorAll('#current-price');
                priceElements.forEach(el => {
                    const oldPrice = parseFloat(el.textContent.replace('$', '').replace(',', ''));
                    el.textContent = '$' + data.price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    
                    // Add pulse animation if price changed
                    if (oldPrice !== data.price) {
                        el.classList.add('price-pulse');
                        setTimeout(() => {
                            el.classList.remove('price-pulse');
                        }, 1000);
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error updating current price:', error);
        });
}

function generatePrediction() {
    // Show loading indicator
    document.getElementById('prediction-intro').classList.add('d-none');
    document.getElementById('prediction-content').classList.add('d-none');
    document.getElementById('prediction-loading').classList.remove('d-none');
    
    // Gather form data
    const formData = {
        histRange: parseInt(document.getElementById('histRange').value),
        predDays: parseInt(document.getElementById('predDays').value),
        targetAccuracy: parseFloat(document.getElementById('targetAccuracy').value),
        useEnsemble: document.getElementById('useEnsemble').checked,
        mlWeight: parseFloat(document.getElementById('mlWeight').value),
        arimaWeight: parseFloat(document.getElementById('arimaWeight').value),
        esWeight: parseFloat(document.getElementById('esWeight').value)
    };
    
    // Update prediction days label
    document.getElementById('pred-days-label').textContent = formData.predDays;
    
    // Make API request
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        updatePredictionResults(data);
        updateAccuracyMetrics(data);
    })
    .catch(error => {
        console.error('Error generating prediction:', error);
        // Show error message
        document.getElementById('prediction-loading').classList.add('d-none');
        document.getElementById('prediction-intro').classList.remove('d-none');
        alert('Error generating prediction. Please try again.');
    });
}

function updatePredictionResults(data) {
    // Hide loading indicator
    document.getElementById('prediction-loading').classList.add('d-none');
    
    // Show prediction content
    const predictionContent = document.getElementById('prediction-content');
    predictionContent.classList.remove('d-none');
    predictionContent.classList.add('fade-in');
    
    // Update current price display
    const currentPriceDisplay = document.getElementById('current-price-display');
    currentPriceDisplay.textContent = '$' + data.current_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    
    // Update current price change
    const currentPriceChange = document.getElementById('current-price-change');
    currentPriceChange.textContent = data.price_change.toFixed(2) + '% from yesterday';
    if (data.price_change > 0) {
        currentPriceChange.classList.add('positive');
        currentPriceChange.classList.remove('negative');
    } else if (data.price_change < 0) {
        currentPriceChange.classList.add('negative');
        currentPriceChange.classList.remove('positive');
    } else {
        currentPriceChange.classList.remove('positive', 'negative');
    }
    
    // Update predicted price display
    const predictedPriceDisplay = document.getElementById('predicted-price-display');
    predictedPriceDisplay.textContent = '$' + data.last_prediction.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    
    // Update predicted price change
    const predictedPriceChange = document.getElementById('predicted-price-change');
    predictedPriceChange.textContent = data.prediction_change.toFixed(2) + '%';
    if (data.prediction_change > 0) {
        predictedPriceChange.classList.add('positive');
        predictedPriceChange.classList.remove('negative');
    } else if (data.prediction_change < 0) {
        predictedPriceChange.classList.add('negative');
        predictedPriceChange.classList.remove('positive');
    } else {
        predictedPriceChange.classList.remove('positive', 'negative');
    }
    
    // Update prediction table
    const predictionTable = document.getElementById('prediction-table');
    predictionTable.innerHTML = '';
    
    data.prediction_table.forEach(prediction => {
        const row = document.createElement('tr');
        row.classList.add('slide-in');
        
        // Add day column
        const dayCell = document.createElement('td');
        dayCell.textContent = prediction.day;
        row.appendChild(dayCell);
        
        // Add date column
        const dateCell = document.createElement('td');
        dateCell.textContent = prediction.date;
        row.appendChild(dateCell);
        
        // Add price column
        const priceCell = document.createElement('td');
        priceCell.textContent = '$' + prediction.predictedPrice.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        priceCell.classList.add('font-weight-bold');
        row.appendChild(priceCell);
        
        // Add change column
        const changeCell = document.createElement('td');
        const changeText = prediction.change.toFixed(2) + '%';
        changeCell.textContent = changeText;
        if (prediction.change > 0) {
            changeCell.classList.add('positive');
        } else if (prediction.change < 0) {
            changeCell.classList.add('negative');
        }
        row.appendChild(changeCell);
        
        predictionTable.appendChild(row);
    });
    
    // Plot prediction chart
    fetch('/api/historical-data?days=30')
        .then(response => response.json())
        .then(histData => {
            plotPredictionChart(histData, data);
        })
        .catch(error => {
            console.error('Error fetching historical data for prediction chart:', error);
        });
}

function updateAccuracyMetrics(data) {
    // Show accuracy content
    document.getElementById('accuracy-loading').classList.add('d-none');
    document.getElementById('accuracy-content').classList.remove('d-none');
    
    // Update accuracy metrics
    const metrics = data.accuracy_metrics;
    
    // Combined accuracy
    const combinedAccuracy = document.getElementById('combined-accuracy');
    combinedAccuracy.textContent = metrics.combined_accuracy.toFixed(2) + '%';
    
    // Accuracy vs target
    const targetAccuracy = parseFloat(document.getElementById('targetAccuracy').value);
    const accuracyVsTarget = document.getElementById('accuracy-vs-target');
    const diff = metrics.combined_accuracy - targetAccuracy;
    accuracyVsTarget.textContent = diff.toFixed(2) + '% vs target';
    if (diff >= 0) {
        accuracyVsTarget.classList.add('positive');
        accuracyVsTarget.classList.remove('negative');
    } else {
        accuracyVsTarget.classList.add('negative');
        accuracyVsTarget.classList.remove('positive');
    }
    
    // Directional accuracy
    const directionalAccuracy = document.getElementById('directional-accuracy');
    directionalAccuracy.textContent = metrics.directional_accuracy.toFixed(2) + '%';
    
    // Model info
    const modelInfo = document.getElementById('model-info');
    modelInfo.innerHTML = '';
    
    if (data.model_weights.ml_ensemble === 1.0) {
        modelInfo.innerHTML = '<p>Using optimized single model prediction</p>';
    } else {
        const modelWeightsHtml = '<p>Using ensemble prediction with the following weights:</p><ul>';
        const weights = Object.entries(data.model_weights).map(([model, weight]) => 
            `<li>${model}: ${(weight * 100).toFixed(1)}%</li>`
        ).join('');
        modelInfo.innerHTML = modelWeightsHtml + weights + '</ul>';
    }
    
    // Technical metrics
    const technicalMetricsList = document.getElementById('technical-metrics-list');
    technicalMetricsList.innerHTML = `
        <li>Mean Absolute Error (MAE): $${metrics.mae.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</li>
        <li>Root Mean Squared Error (RMSE): $${metrics.rmse.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</li>
        <li>Price Accuracy: ${metrics.price_accuracy.toFixed(2)}%</li>
    `;
    
    // Plot accuracy chart
    plotAccuracyChart(metrics);
}

function plotHistoricalChart(data) {
    const showVolume = document.getElementById('show-volume').checked;
    const showMA = document.getElementById('show-ma').checked;
    
    // Create traces
    const traces = [];
    
    // Price trace
    traces.push({
        x: data.dates,
        y: data.prices,
        type: 'scatter',
        mode: 'lines',
        name: 'Bitcoin Price',
        line: {
            color: '#F7931A',
            width: 2
        }
    });
    
    // Add volume if requested
    if (showVolume && data.volumes && data.volumes.length > 0) {
        traces.push({
            x: data.dates,
            y: data.volumes,
            type: 'bar',
            name: 'Trading Volume',
            marker: {
                color: 'rgba(200, 200, 200, 0.5)'
            },
            opacity: 0.5,
            yaxis: 'y2'
        });
    }
    
    // Add moving averages if requested and available
    if (showMA && data.show_ma) {
        const ma_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
        const ma_periods = [5, 7, 14, 30];
        
        ma_periods.forEach((period, i) => {
            const maKey = `SMA${period}`;
            if (data[maKey]) {
                traces.push({
                    x: data.dates,
                    y: data[maKey],
                    type: 'scatter',
                    mode: 'lines',
                    name: `${maKey} Moving Average`,
                    line: {
                        color: ma_colors[i % ma_colors.length],
                        width: 1.5,
                        dash: 'dot'
                    },
                    opacity: 0.7
                });
            }
        });
    }
    
    // Layout configuration
    let layout = {
        title: 'Bitcoin Price History',
        xaxis: {
            title: 'Date',
            rangeslider: { visible: true },
            rangeselector: {
                buttons: [
                    { count: 7, label: '1w', step: 'day', stepmode: 'backward' },
                    { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
                    { count: 3, label: '3m', step: 'month', stepmode: 'backward' },
                    { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
                    { step: 'all', label: 'All' }
                ]
            }
        },
        yaxis: {
            title: 'Price (USD)',
            tickprefix: '$'
        },
        legend: {
            orientation: 'h',
            y: 1.1
        },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    };
    
    // Add second y-axis for volume if needed
    if (showVolume) {
        layout.yaxis2 = {
            title: 'Volume (USD)',
            overlaying: 'y',
            side: 'right',
            showgrid: false
        };
    }
    
    // Create the plot
    Plotly.newPlot('historical-chart', traces, layout);
    
    // Add recent prices to the table
    if (data.dates && data.dates.length > 0) {
        const recentDates = data.dates.slice(-10);
        const recentPrices = data.prices.slice(-10);
        const recentVolumes = data.volumes ? data.volumes.slice(-10) : [];
        
        const historyTable = [];
        for (let i = 0; i < recentDates.length; i++) {
            historyTable.push({
                date: recentDates[i],
                price: recentPrices[i],
                volume: recentVolumes[i] || 0
            });
        }
        
        populateHistoricalTable(historyTable);
    }
}

function plotPredictionChart(histData, predData) {
    // Extract historical data
    const historicalDates = histData.dates.slice(-30);  // Last 30 days
    const historicalPrices = histData.prices.slice(-30);
    
    // Extract prediction data
    const futureDates = predData.future_dates;
    const predictedPrices = predData.predicted_prices;
    
    // Create confidence intervals
    const last_price = historicalPrices[historicalPrices.length - 1];
    const std_dev = calculateStdDev(historicalPrices) * 0.1;
    
    const upper_bound = predictedPrices.map((price, i) => price + std_dev * (i+1));
    const lower_bound = predictedPrices.map((price, i) => price - std_dev * (i+1));
    
    // Create traces
    const traces = [];
    
    // Historical price trace
    traces.push({
        x: historicalDates,
        y: historicalPrices,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Price',
        line: {
            color: '#1f77b4',
            width: 2
        }
    });
    
    // Prediction line
    traces.push({
        x: futureDates,
        y: predictedPrices,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Price',
        line: {
            color: '#F7931A',
            width: 3,
            dash: 'dot'
        },
        marker: {
            size: 8,
            symbol: 'diamond'
        }
    });
    
    // Confidence interval - upper bound
    traces.push({
        x: futureDates,
        y: upper_bound,
        type: 'scatter',
        mode: 'lines',
        name: 'Upper Bound',
        line: {
            width: 0
        },
        marker: {
            color: "#F7931A"
        },
        showlegend: false
    });
    
    // Confidence interval - lower bound
    traces.push({
        x: futureDates,
        y: lower_bound,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower Bound',
        line: {
            width: 0
        },
        marker: {
            color: "#F7931A"
        },
        fillcolor: 'rgba(247, 147, 26, 0.2)',
        fill: 'tonexty',
        showlegend: false
    });
    
    // Add connecting line between last historical price and first prediction
    traces.push({
        x: [historicalDates[historicalDates.length - 1], futureDates[0]],
        y: [historicalPrices[historicalPrices.length - 1], predictedPrices[0]],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#F7931A',
            width: 1.5,
            dash: 'dot'
        },
        showlegend: false
    });
    
    // Layout configuration
    const layout = {
        title: 'Bitcoin Price Prediction',
        xaxis: {
            title: 'Date'
        },
        yaxis: {
            title: 'Price (USD)',
            tickprefix: '$'
        },
        legend: {
            orientation: 'h',
            y: 1.1
        },
        margin: { l: 50, r: 50, t: 50, b: 50 },
        shapes: [
            // Add vertical line at prediction start
            {
                type: "line",
                x0: futureDates[0],
                y0: 0,
                x1: futureDates[0],
                y1: 1,
                xref: 'x',
                yref: 'paper',
                line: {
                    color: "gray",
                    width: 1,
                    dash: "dash"
                }
            }
        ],
        annotations: [
            {
                x: futureDates[0],
                y: 1,
                xref: 'x',
                yref: 'paper',
                text: 'Prediction Start',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 1,
                arrowcolor: "gray",
                ax: 50,
                ay: -30
            }
        ]
    };
    
    // Create the plot
    Plotly.newPlot('prediction-chart', traces, layout);
}

function plotAccuracyChart(metrics) {
    // Create gauge chart for overall accuracy
    const traces = [
        {
            type: "indicator",
            mode: "gauge+number",
            value: metrics.combined_accuracy,
            title: { text: "Prediction Accuracy" },
            gauge: {
                axis: { range: [0, 100], tickwidth: 1, tickcolor: "darkblue" },
                bar: { color: "darkblue" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 60], color: 'red' },
                    { range: [60, 80], color: 'orange' },
                    { range: [80, 90], color: 'yellow' },
                    { range: [90, 100], color: 'green' }
                ],
                threshold: {
                    line: { color: "red", width: 4 },
                    thickness: 0.75,
                    value: 95
                }
            },
            domain: { row: 0, column: 0 }
        },
        {
            type: "indicator",
            mode: "gauge+number",
            value: metrics.directional_accuracy,
            title: { text: "Directional Accuracy" },
            gauge: {
                axis: { range: [0, 100], tickwidth: 1, tickcolor: "darkblue" },
                bar: { color: "darkblue" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 60], color: 'red' },
                    { range: [60, 80], color: 'orange' },
                    { range: [80, 90], color: 'yellow' },
                    { range: [90, 100], color: 'green' }
                ]
            },
            domain: { row: 0, column: 1 }
        },
        {
            type: "indicator",
            mode: "gauge+number",
            value: metrics.price_accuracy,
            title: { text: "Price Accuracy" },
            gauge: {
                axis: { range: [0, 100], tickwidth: 1, tickcolor: "darkblue" },
                bar: { color: "darkblue" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 60], color: 'red' },
                    { range: [60, 80], color: 'orange' },
                    { range: [80, 90], color: 'yellow' },
                    { range: [90, 100], color: 'green' }
                ]
            },
            domain: { row: 1, column: 0 }
        },
        {
            type: "indicator",
            mode: "number+delta",
            value: metrics.mae,
            title: { text: "Mean Absolute Error" },
            delta: { reference: metrics.rmse, relative: true },
            domain: { row: 1, column: 1 }
        }
    ];
    
    const layout = {
        grid: { rows: 2, columns: 2, pattern: "independent" },
        margin: { t: 30, b: 30, l: 30, r: 30 }
    };
    
    Plotly.newPlot('accuracy-chart', traces, layout);
}

function populateHistoricalTable(data) {
    const tableBody = document.getElementById('historical-table');
    tableBody.innerHTML = '';
    
    // If data is an object with dates property, use it directly
    let tableData = [];
    if (Array.isArray(data)) {
        tableData = data;
    } else if (data.dates && data.prices) {
        const len = Math.min(10, data.dates.length);
        for (let i = data.dates.length - len; i < data.dates.length; i++) {
            tableData.push({
                date: data.dates[i],
                price: data.prices[i],
                volume: data.volumes ? data.volumes[i] : 0
            });
        }
    }
    
    tableData.forEach(entry => {
        const row = document.createElement('tr');
        
        // Date
        const dateCell = document.createElement('td');
        dateCell.textContent = entry.date;
        row.appendChild(dateCell);
        
        // Price
        const priceCell = document.createElement('td');
        priceCell.textContent = '$' + entry.price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        row.appendChild(priceCell);
        
        // Volume
        const volumeCell = document.createElement('td');
        volumeCell.textContent = '$' + entry.volume.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
        row.appendChild(volumeCell);
        
        tableBody.appendChild(row);
    });
}

function updateHistoricalChart() {
    const showVolume = document.getElementById('show-volume').checked;
    const showMA = document.getElementById('show-ma').checked;
    
    fetch(`/api/historical-data?show_volume=${showVolume}&show_ma=${showMA}`)
        .then(response => response.json())
        .then(data => {
            plotHistoricalChart(data);
        })
        .catch(error => {
            console.error('Error updating historical chart:', error);
        });
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    
    // Update any charts that are already rendered
    const charts = ['historical-chart', 'prediction-chart', 'accuracy-chart'];
    charts.forEach(chart => {
        const chartElement = document.getElementById(chart);
        if (chartElement && chartElement.data) {
            Plotly.relayout(chart, {
                paper_bgcolor: getComputedStyle(document.body).getPropertyValue('--bg-color'),
                plot_bgcolor: getComputedStyle(document.body).getPropertyValue('--bg-color'),
                font: {
                    color: getComputedStyle(document.body).getPropertyValue('--text-color')
                }
            });
        }
    });
}

function calculateStdDev(values) {
    const n = values.length;
    if (n <= 1) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / n;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
    return Math.sqrt(variance);
}
