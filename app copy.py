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
        
        return df_expanded
    
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty")
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Error details:", str(e.__class__.__name__))
        return None

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
                        predictions = process_and_predict(uploaded_file, model)
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
                                    
                                    # Create tabs for different visualizations
                                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Time Series Plot", "Daily Distribution", "Summary Statistics"])
                                    
                                    with viz_tab1:
                                        st.write("#### Time Series Forecast")
                                        fig1 = plt.figure(figsize=(12, 6))
                                        plt.plot(ts_data['predicted_date'], ts_data['predictions'], 
                                                marker='o', linestyle='-', linewidth=2, markersize=8)
                                        plt.title(f'Sales Forecast for {selected_ts}')
                                        plt.xlabel('Date')
                                        plt.ylabel('Predicted Sales')
                                        plt.xticks(rotation=45)
                                        plt.grid(True)
                                        st.pyplot(fig1)
                                        plt.close()
                                    
                                    with viz_tab2:
                                        st.write("#### Daily Sales Distribution")
                                        fig2 = plt.figure(figsize=(10, 6))
                                        plt.hist(ts_data['predictions'], bins=15, edgecolor='black')
                                        plt.title(f'Distribution of Predicted Sales for {selected_ts}')
                                        plt.xlabel('Predicted Sales')
                                        plt.ylabel('Frequency')
                                        plt.grid(True)
                                        st.pyplot(fig2)
                                        plt.close()
                                    
                                    with viz_tab3:
                                        st.write("#### Summary Statistics")
                                        col1, col2 = st.columns(2)
                                        
                                        # Calculate statistics
                                        stats = {
                                            "Mean Predicted Sales": ts_data['predictions'].mean(),
                                            "Maximum Prediction": ts_data['predictions'].max(),
                                            "Minimum Prediction": ts_data['predictions'].min(),
                                            "Standard Deviation": ts_data['predictions'].std(),
                                            "Total Predicted Sales": ts_data['predictions'].sum()
                                        }
                                        
                                        # Display metrics
                                        for i, (metric, value) in enumerate(stats.items()):
                                            if i % 2 == 0:
                                                col1.metric(metric, f"{value:.2f}")
                                            else:
                                                col2.metric(metric, f"{value:.2f}")
                                    
                                        # Add comparison feature
                                        st.write("### Compare Multiple Time Series")
                                        selected_compare = st.multiselect(
                                            "Select additional Store-Family combinations to compare:",
                                            options=[ts for ts in predictions['time_series_id'].unique() if ts != selected_ts],
                                            max_selections=3
                                        )
                                        
                                        if selected_compare:
                                            fig3 = plt.figure(figsize=(12, 6))
                                            
                                            # Plot main selection
                                            plt.plot(ts_data['predicted_date'], ts_data['predictions'], 
                                                    marker='o', label=selected_ts)
                                            
                                            # Plot comparisons
                                            colors = ['red', 'green', 'blue']
                                            for i, ts in enumerate(selected_compare):
                                                compare_data = predictions[predictions['time_series_id'] == ts]
                                                plt.plot(compare_data['predicted_date'], compare_data['predictions'], 
                                                        marker='o', label=ts, color=colors[i])
                                            
                                            plt.title('Sales Forecast Comparison')
                                            plt.xlabel('Date')
                                            plt.ylabel('Predicted Sales')
                                            plt.xticks(rotation=45)
                                            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                            plt.grid(True)
                                            st.pyplot(fig3)
                                            plt.close()
                                            
                                            # Add summary comparison table
                                            st.write("### Comparison Summary")
                                            compare_df = pd.DataFrame()
                                            compare_df['Store-Family'] = [selected_ts] + selected_compare
                                            compare_df['Average Sales'] = [predictions[predictions['time_series_id'] == ts]['predictions'].mean() 
                                                                         for ts in compare_df['Store-Family']]
                                            compare_df['Total Sales'] = [predictions[predictions['time_series_id'] == ts]['predictions'].sum() 
                                                                       for ts in compare_df['Store-Family']]
                                            compare_df = compare_df.round(2)
                                            st.dataframe(compare_df)
                                    
                                    # Add download options for visualizations
                                    st.write("### Download Options")
                                    if st.button("Download All Visualizations"):
                                        # Create PDF with all visualizations
                                        buffer = io.BytesIO()
                                        pdf = PdfPages(buffer)
                                        
                                        # Save all figures to PDF
                                        for fig in [fig1, fig2, fig3]:
                                            pdf.savefig(fig)
                                        pdf.close()
                                        
                                        # Offer download
                                        st.download_button(
                                            label="Download Visualizations (PDF)",
                                            data=buffer.getvalue(),
                                            file_name="predictions_visualizations.pdf",
                                            mime="application/pdf"
                                        )
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                # st.write("Please ensure the model file 'time_series_model.h5' is in the same directory as this script.")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.write("Error details:", str(e.__class__.__name__))

if __name__ == "__main__":
    main()
