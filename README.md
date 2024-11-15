# Time Series Sales Forecasting Model & Web Interface

This project consists of a deep learning model for sales forecasting and a user-friendly web interface built with Streamlit. The model predicts future sales for different store-family combinations based on historical data.

## Model Architecture

The model uses a hybrid CNN-LSTM architecture:
- Input Size: 384 time steps
- Output Size: 16 time steps (16-day forecast)
- Architecture:
  - Multiple Convolutional layers with dilated causal convolutions
  - Dropout layer (0.3)
  - LSTM layer
  - Dense output layer

### Model Performance
- Loss Function: Mean Squared Error
- Custom Metric: Root Mean Squared Logarithmic Error (RMSLE)
- Training includes early stopping and learning rate reduction

## Web Interface Features

### Data Upload
- Accepts CSV files with required columns:
  - store_nbr
  - family
  - date
  - sales

### Predictions
- Generates 16-day sales forecasts
- Displays predictions in an organized table
- Shows predicted dates and values
- Downloadable results in CSV format

### Visualizations
1. Forecast vs History
   - Historical sales data comparison
   - Trend visualization

2. Sales Patterns
   - Distribution comparison histograms
   - Box plots for statistical comparison

3. Statistical Analysis
   - Detailed metrics comparison
   - Visual representation of key statistics

4. Trend Analysis
   - Moving averages (7-day and 30-day)
   - Long-term trend visualization

5. Comparative Analysis
   - Multi-series comparison
   - Detailed metrics table

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/Jay254p/Time-Series-Forecasting
cd https://github.com/Jay254p/Time-Series-Forecasting
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your trained model file:
- Save your model as 'time_series_model.h5'
- Place it in the same directory as app.py

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage Guide

1. Data Preparation:
   - Ensure your CSV file has the required columns
   - Data should be in daily format
   - Sort by store_nbr, family, and date

2. Making Predictions:
   - Upload your CSV file through the web interface
   - Click "Make Predictions"
   - View results in the Predictions tab

3. Analyzing Results:
   - Use the Visualizations tab to explore predictions
   - Compare different store-family combinations
   - Download visualizations as PDF

## Requirements

```txt
tensorflow>=2.0.0
streamlit>=1.0.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## File Structure

```
├── app.py              # Streamlit web interface
├── requirements.txt    # Package dependencies
└── time_series_model.h5# Trained model file
```

## Input Data Format

Example CSV structure:
```csv
date,store_nbr,family,sales
2023-01-01,1,GROCERY,100.5
2023-01-01,1,BEVERAGES,50.2
...
```

## Output Format

The model generates:
1. Daily sales predictions for the next 16 days
2. Prediction confidence metrics
3. Visualization of trends and patterns

## Contributing

Feel free to:
- Open issues
- Submit pull requests
- Suggest improvements
- Report bugs


This README provides:
1. Clear project overview
2. Installation instructions
3. Usage guidelines
4. Technical details
5. File structure
6. Input/Output formats


