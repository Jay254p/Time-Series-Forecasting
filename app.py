import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Constants
INPUT_SIZE = 384
OUTPUT_SIZE = 16
SHIFT = 1
BATCH_SIZE = 64

class RMSLE(tf.keras.metrics.Metric):
    def __init__(self, name: str = "rmsle", **kwargs) -> None:
        super(RMSLE, self).__init__(name=name, **kwargs)
        self.sum_squared_log_error = self.add_weight(name='sum_squared_log_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        log_error = tf.math.log1p(y_pred) - tf.math.log1p(y_true)
        squared_log_error = tf.square(log_error)
        self.sum_squared_log_error.assign_add(tf.reduce_sum(squared_log_error))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self) -> tf.Tensor:
        mean_squared_log_error = self.sum_squared_log_error / self.count
        return tf.sqrt(mean_squared_log_error)

    def reset_states(self) -> None:
        self.sum_squared_log_error.assign(0.0)
        self.count.assign(0.0)

def plot_predictions(original_data, predictions, time_series_id):
    """Create a plot comparing original data with predictions"""
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data', marker='o')
    plt.plot(range(len(original_data), len(original_data) + len(predictions)), 
             predictions, label='Predictions', marker='o')
    plt.title(f'Time Series Predictions for {time_series_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    return plt

def process_and_predict(uploaded_file, model):
    """Process uploaded file and make predictions"""
    try:
        # Debug information
        st.write("File details:")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")

        # Read and process the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Check if DataFrame is empty
        if df.empty:
            st.error("The uploaded file is empty")
            return None
            
        # Check required columns
        required_columns = ['store_nbr', 'family', 'date', 'sales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.write("Available columns:", df.columns.tolist())
            return None

        # Display DataFrame info
        st.write("DataFrame Info:")
        st.write(f"Shape: {df.shape}")
        st.write("Columns:", df.columns.tolist())
        st.write("First few rows:")
        st.dataframe(df.head())

        # Continue with processing
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        df['time_series_id'] = df['store_nbr'].astype(str) + '_' + df['family']
        
        # Pivot the data
        pivoted_df = df.pivot_table(index='time_series_id', columns='date', values='sales', fill_value=0)
        series_array = pivoted_df.to_numpy()
        
        # Scale the data
        scaler = MinMaxScaler()
        series_array = scaler.fit_transform(series_array.T).T
        
        # Make predictions
        idx_list = pivoted_df.index.tolist()
        nb_ts = len(idx_list)
        res = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (_, idx) in enumerate(zip(series_array, idx_list)):
            to_predict = series_array[i][-INPUT_SIZE:].reshape(1, -1)
            res.append(model.predict(to_predict, verbose=0).tolist()[0])
            
            # Update progress
            progress = (i + 1) / nb_ts
            progress_bar.progress(progress)
            status_text.text(f"Processing... {i+1}/{nb_ts} time series")
        
        preds = np.array(res)
        preds = scaler.inverse_transform(preds.T).T
        
        # Get the last date from the input data
        last_date = pd.to_datetime(df['date']).max()
        
        # Format predictions
        df_preds = pd.DataFrame(preds, index=pd.Index(idx_list, name="time_series_id"))
        df_stacked = pd.DataFrame({
            'time_series_id': df_preds.index,
            'predictions': df_preds.apply(lambda row: row.values.tolist(), axis=1)
        })
        
        df_expanded = df_stacked.explode('predictions').reset_index(drop=True)
        df_expanded['prediction_order'] = df_expanded.groupby('time_series_id').cumcount()
        
        # Add predicted dates
        df_expanded['predicted_date'] = df_expanded['prediction_order'].apply(
            lambda x: last_date + pd.Timedelta(days=x+1)
        )
        
        # Reorder columns for clarity
        df_expanded = df_expanded[['time_series_id', 'predicted_date', 'predictions', 'prediction_order']]
        
        # Return both predictions and original data
        return df_expanded, df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

def load_model_with_metrics():
    """Load the model and initialize metrics properly"""
    try:
        # Define custom objects
        custom_objects = {
            'RMSLE': RMSLE,
            'loss': tf.keras.losses.MeanSquaredError(),
            'MeanSquaredError': tf.keras.losses.MeanSquaredError
        }
        
        # Load model with custom objects
        model = tf.keras.models.load_model('time_series_model_V3.h5', custom_objects=custom_objects)
        
        # Recompile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[RMSLE()]
        )
        
        # Initialize metrics with dummy data
        dummy_input = np.zeros((1, INPUT_SIZE))
        model.predict(dummy_input, verbose=0)
        
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def main():
    st.set_page_config(page_title="Time Series Prediction App", layout="wide")
    
    st.title("Time Series Prediction App")
    
    # Add sidebar with information
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses a trained deep learning model to make time series predictions.
    
    The model expects data with the following columns:
    - store_nbr
    - family
    - date
    - sales
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Try to read the first few rows to validate the file
            df_preview = pd.read_csv(uploaded_file)
            if df_preview.empty:
                st.error("The uploaded file is empty")
                return
                
            uploaded_file.seek(0)
            st.write("### Preview of uploaded data:")
            st.dataframe(df_preview.head())
            
            # Load model when needed
            try:
                model = load_model_with_metrics()
                
                if st.button('Make Predictions'):
                    with st.spinner('Processing data...'):
                        predictions, original_df = process_and_predict(uploaded_file, model)
                        if predictions is not None:
                            st.success("Predictions completed!")
                            
                            # Create tabs for different views
                            tab1, tab2 = st.tabs(["Predictions", "Visualizations"])
                            
                            with tab1:
                                st.write("### Preview of predictions:")
                                
                                # Format the predictions table
                                display_df = predictions.copy()
                                
                                # Round predictions to 1 decimal place
                                display_df['predictions'] = display_df['predictions'].round(1)
                                
                                # Format the date
                                display_df['predicted_date'] = pd.to_datetime(display_df['predicted_date']).dt.strftime('%Y-%m-%d')
                                
                                # Display with custom formatting
                                st.dataframe(
                                    display_df,
                                    column_config={
                                        "time_series_id": "Store-Family",
                                        "predicted_date": "Date",
                                        "predictions": "Sales Prediction",
                                        "prediction_order": "Day Order"
                                    },
                                    hide_index=True,
                                )
                                
                                # Download button
                                csv = display_df.to_csv(index=False)
                                st.download_button(
                                    label="Download predictions as CSV",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                            
                            with tab2:
                                st.write("### Visualization of Predictions")
                                
                                # 1. Select time series to visualize
                                selected_ts = st.selectbox(
                                    "Select Store-Family combination:",
                                    options=predictions['time_series_id'].unique()
                                )
                                
                                if selected_ts:
                                    # Filter data for selected time series
                                    ts_data = predictions[predictions['time_series_id'] == selected_ts]
                                    
                                    # Get historical data for comparison
                                    historical_data = original_df[original_df['time_series_id'] == selected_ts].copy()
                                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                                    
                                    # Create tabs for different visualizations
                                    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
                                        "Forecast vs History", 
                                        "Sales Patterns", 
                                        "Statistical Analysis",
                                        "Trend Analysis",
                                        "Comparative Analysis"
                                    ])
                                    
                                    # Store figures for later download
                                    figures = []
                                    
                                    with viz_tab1:
                                        st.write("#### Historical Data vs Forecast")
                                        fig1 = plt.figure(figsize=(15, 7))
                                        
                                        # Plot historical data
                                        plt.plot(historical_data['date'], historical_data['sales'], 
                                                label='Historical Sales', alpha=0.7)
                                        
                                        # Plot predictions
                                        plt.plot(ts_data['predicted_date'], ts_data['predictions'], 
                                                label='Forecasted Sales', linestyle='--', marker='o')
                                        
                                        plt.title(f'Historical and Forecasted Sales for {selected_ts}')
                                        plt.xlabel('Date')
                                        plt.ylabel('Sales')
                                        plt.xticks(rotation=45)
                                        plt.legend()
                                        plt.grid(True)
                                        figures.append(fig1)
                                        st.pyplot(fig1)
                                        plt.close()

                                    with viz_tab2:
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Sales Distribution Comparison
                                            fig2a = plt.figure(figsize=(10, 6))
                                            plt.hist(historical_data['sales'], bins=30, alpha=0.5, 
                                                    label='Historical', color='blue')
                                            plt.hist(ts_data['predictions'], bins=15, alpha=0.5, 
                                                    label='Predicted', color='red')
                                            plt.title('Sales Distribution Comparison')
                                            plt.xlabel('Sales Value')
                                            plt.ylabel('Frequency')
                                            plt.legend()
                                            figures.append(fig2a)
                                            st.pyplot(fig2a)
                                            plt.close()

                                        with col2:
                                            # Box Plot Comparison
                                            fig2b = plt.figure(figsize=(10, 6))
                                            data_to_plot = [historical_data['sales'], ts_data['predictions']]
                                            plt.boxplot(data_to_plot, labels=['Historical', 'Predicted'])
                                            plt.title('Sales Distribution Box Plot')
                                            plt.ylabel('Sales Value')
                                            figures.append(fig2b)
                                            st.pyplot(fig2b)
                                            plt.close()

                                    with viz_tab3:
                                        st.write("#### Statistical Analysis")
                                        
                                        # Calculate statistics safely
                                        hist_stats = pd.Series({
                                            'mean': historical_data['sales'].mean(),
                                            'std': historical_data['sales'].std(),
                                            'min': historical_data['sales'].min(),
                                            'max': historical_data['sales'].max(),
                                            'median': historical_data['sales'].median()
                                        })
                                        
                                        pred_stats = pd.Series({
                                            'mean': ts_data['predictions'].mean(),
                                            'std': ts_data['predictions'].std(),
                                            'min': ts_data['predictions'].min(),
                                            'max': ts_data['predictions'].max(),
                                            'median': ts_data['predictions'].median()
                                        })
                                        
                                        # Create comparison DataFrame
                                        stats_comparison = pd.DataFrame({
                                            'Historical': hist_stats,
                                            'Predicted': pred_stats
                                        }).round(2)
                                        
                                        st.dataframe(stats_comparison)
                                        
                                        # Visualization of key metrics
                                        fig3 = plt.figure(figsize=(12, 6))
                                        metrics = ['mean', 'std', 'min', 'max']
                                        x = range(len(metrics))
                                        width = 0.35
                                        
                                        plt.bar([i - width/2 for i in x], 
                                               [hist_stats[m] for m in metrics], 
                                               width, label='Historical', alpha=0.8)
                                        plt.bar([i + width/2 for i in x], 
                                               [pred_stats[m] for m in metrics], 
                                               width, label='Predicted', alpha=0.8)
                                        
                                        plt.xticks(x, metrics)
                                        plt.title('Key Metrics Comparison')
                                        plt.legend()
                                        figures.append(fig3)
                                        st.pyplot(fig3)
                                        plt.close()

                                    with viz_tab4:
                                        st.write("#### Trend Analysis")
                                        
                                        # Moving averages for historical data
                                        historical_data['MA7'] = historical_data['sales'].rolling(7).mean()
                                        historical_data['MA30'] = historical_data['sales'].rolling(30).mean()
                                        
                                        fig4 = plt.figure(figsize=(15, 7))
                                        plt.plot(historical_data['date'], historical_data['sales'], 
                                                label='Daily Sales', alpha=0.4)
                                        plt.plot(historical_data['date'], historical_data['MA7'], 
                                                label='7-day Moving Average')
                                        plt.plot(historical_data['date'], historical_data['MA30'], 
                                                label='30-day Moving Average')
                                        plt.plot(ts_data['predicted_date'], ts_data['predictions'], 
                                                label='Predictions', linestyle='--')
                                        
                                        plt.title('Sales Trend Analysis')
                                        plt.xlabel('Date')
                                        plt.ylabel('Sales')
                                        plt.legend()
                                        plt.grid(True)
                                        figures.append(fig4)
                                        st.pyplot(fig4)
                                        plt.close()

                                    with viz_tab5:
                                        st.write("### Comparative Analysis")
                                        
                                        # Allow comparison with other store-families
                                        selected_compare = st.multiselect(
                                            "Select store-families to compare:",
                                            options=[ts for ts in predictions['time_series_id'].unique() 
                                                    if ts != selected_ts],
                                            max_selections=3
                                        )
                                        
                                        if selected_compare:
                                            fig5 = plt.figure(figsize=(15, 7))
                                            
                                            # Plot main selection
                                            plt.plot(ts_data['predicted_date'], ts_data['predictions'], 
                                                    marker='o', label=selected_ts)
                                            
                                            # Plot comparisons
                                            colors = ['red', 'green', 'blue']
                                            for i, ts in enumerate(selected_compare):
                                                compare_data = predictions[predictions['time_series_id'] == ts]
                                                plt.plot(compare_data['predicted_date'], 
                                                        compare_data['predictions'], 
                                                        marker='o', label=ts, color=colors[i])
                                            
                                            plt.title('Multi-Series Forecast Comparison')
                                            plt.xlabel('Date')
                                            plt.ylabel('Predicted Sales')
                                            plt.xticks(rotation=45)
                                            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                            plt.grid(True)
                                            figures.append(fig5)
                                            st.pyplot(fig5)
                                            plt.close()
                                            
                                            # Comparison summary table
                                            st.write("#### Comparison Summary")
                                            compare_df = pd.DataFrame()
                                            compare_df['Store-Family'] = [selected_ts] + selected_compare
                                            compare_df['Mean Sales'] = [
                                                predictions[predictions['time_series_id'] == ts]['predictions'].mean() 
                                                for ts in compare_df['Store-Family']
                                            ]
                                            compare_df['Total Sales'] = [
                                                predictions[predictions['time_series_id'] == ts]['predictions'].sum() 
                                                for ts in compare_df['Store-Family']
                                            ]
                                            compare_df['Min Sales'] = [
                                                predictions[predictions['time_series_id'] == ts]['predictions'].min() 
                                                for ts in compare_df['Store-Family']
                                            ]
                                            compare_df['Max Sales'] = [
                                                predictions[predictions['time_series_id'] == ts]['predictions'].max() 
                                                for ts in compare_df['Store-Family']
                                            ]
                                            compare_df = compare_df.round(2)
                                            st.dataframe(compare_df)

                                        # Download options
                                        st.write("### Download Visualizations")
                                        if st.button("Download All Visualizations"):
                                            try:
                                                # Create PDF with all visualizations
                                                buffer = io.BytesIO()
                                                pdf = PdfPages(buffer)
                                                
                                                # Save all figures to PDF
                                                for fig in figures:
                                                    pdf.savefig(fig)
                                                pdf.close()
                                                
                                                # Offer download
                                                st.download_button(
                                                    label="Download Visualizations (PDF)",
                                                    data=buffer.getvalue(),
                                                    file_name=f"predictions_visualizations_{selected_ts}.pdf",
                                                    mime="application/pdf"
                                                )
                                                
                                                st.success("PDF generated successfully!")
                                            except Exception as e:
                                                st.error(f"Error generating PDF: {str(e)}")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                # st.write("Please ensure the model file 'time_series_model.h5' is in the same directory as this script.")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.write("Error details:", str(e.__class__.__name__))

if __name__ == "__main__":
    main()
