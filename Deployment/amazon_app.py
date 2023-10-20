import streamlit as st
import statsmodels.api as sm
import base64
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
from PIL import Image
import requests

#------------------------------------------------------------------------------#
# Page setup - the page title and layout
im = Image.open('icon.png')
st.set_page_config(page_title='The BullBear Oracle', page_icon=im,
    layout='wide', initial_sidebar_state="expanded",menu_items={
        'Get Help': 'mailto:waruchu.analyst@gmail.com',
        'Report a bug': "mailto:waruchu.analyst@gmail.com",
        'About': "# The BullBears at Moringa School created this app. We hope it helps!"
    })
st.sidebar.subheader("About")
st.sidebar.write("The BullBears at Moringa School created this app. It uses pre-determined parameters of a SARIMA model to predict stock prices.")
st.sidebar.write("We hope it helps!")

# Read and inject CSS
# with open("style.css") as f:
#     css = f.read()
# st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
# st.markdown('<link rel="stylesheet" href="path/to/style.css" type="text/css">',
#     unsafe_allow_html=True)

st.title('**Welcome!**')
# st.markdown('<p class="header">Welcome!</p>', unsafe_allow_html=True)
st.write('This app displays stock price predictions.') 

# Set defaults
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/kuriawaruchu/Amazon/main/main_data_diff_df.csv"
DEFAULT_PREDICTIONS_URL = "https://raw.githubusercontent.com/kuriawaruchu/Amazon/main/stationary_predictions.csv"
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_SARIMA_HYPERPARAMETERS = {
    "p": 2,
    "d": 0,
    "q": 2,
    "seasonal_p": 2,
    "seasonal_d": 0,
    "seasonal_q": 2,
    "s": 5,
}
# Read and inject CSS
with open("style-rough.css") as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Preparing raw file for the model
def generate_sarima_predictions(start_date, end_date, sarima_hyperparameters):
    # Load the historical data
    df = pd.read_csv(DEFAULT_CSV_URL)
    
    # Split the data into X and y variables
    target_column = st.text_input("Specify the target column name as it is written in your file:", "Adj Close")
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    # Split the data into train and test sets
    train_size = DEFAULT_TRAIN_SIZE
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    
    # Fit the SARIMA model
    sarima_model = SARIMAX(y_train, order=(sarima_hyperparameters["p"], sarima_hyperparameters["d"], sarima_hyperparameters["q"]),
                          seasonal_order=(sarima_hyperparameters["seasonal_p"], sarima_hyperparameters["seasonal_d"],
                                          sarima_hyperparameters["seasonal_q"], sarima_hyperparameters["s"]))
    sarima_results = sarima_model.fit()
    
    # Generate predictions for the specified date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    predictions = sarima_results.predict(start=len(y_train), end=len(y_train) + len(date_range) - 1, dynamic=False)
    
    # Create a DataFrame with the predictions and date range
    predictions_df = pd.DataFrame({'Datetime': date_range, 'Predicted Stock Price': predictions})
    
    return predictions_df

def main():
    st.sidebar.image("https://github.com/kuriawaruchu/Amazon/blob/main/logo4.png?raw=true", width="100%", use_column_width=True)
    
     # Generate predictions using the SARIMA model
    st.subheader('Choose Dates')
    start_date = st.date_input("Start date:", pd.to_datetime("2023-10-9"))
    end_date = st.date_input("End date:", pd.to_datetime("2023-11-8"))
    sarima_hyperparameters = DEFAULT_SARIMA_HYPERPARAMETERS
    
    # Allow the user to upload a predictions CSV file
    st.sidebar.header('Fetching your Data')
    st.sidebar.subheader('Upload Predictions')
    predictions_csv_file = st.sidebar.file_uploader("Upload predictions CSV file:", type=["csv"])
    st.markdown('<style>.fileinput-button { background-color: #007BFF; color: #fff; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }</style>', unsafe_allow_html=True)

    predictions_csv_link = st.sidebar.text_input("Or paste a link to predictions CSV file (Enter to apply):")
    
    if predictions_csv_file:
        predictions_df = pd.read_csv(predictions_csv_file)
    elif predictions_csv_link:
        predictions_df = pd.read_csv(predictions_csv_link)
    else:
        # Load the default predictions from the URL
        predictions_df = pd.read_csv(DEFAULT_PREDICTIONS_URL)

    if predictions_df is not None:
        st.subheader('Viewing Predictions')
        
        # Filter predictions based on the selected date range
        predictions_df['Datetime'] = pd.to_datetime(predictions_df['Datetime'])
        mask = (predictions_df['Datetime'] >= pd.to_datetime(start_date)) & (predictions_df['Datetime'] <= pd.to_datetime(end_date))
        predictions_df = predictions_df.loc[mask]
        
        # Allow the user to dynamically change the target column name
        target_column = st.text_input("Specify the target column name as it is written in your file:", "Predicted_Adj_Close")
        
        if target_column not in predictions_df.columns:
            st.warning(f"Column name '{target_column}' not found in the predictions CSV.")
        else:
            # Display the predictions
            st.table(predictions_df)
            st.subheader(f'Predictions graph for {start_date} to {end_date}')
            st.line_chart(predictions_df.set_index('Datetime')[target_column])     
        
       # Allow the user to upload a raw data CSV file
    st.sidebar.subheader('Upload CSV File for training')
    raw_data_csv_file = st.sidebar.file_uploader("Upload CSV file for training:", type=["csv"])
    raw_data_csv_link = st.sidebar.text_input("Or paste a link to the raw data CSV file (Enter to apply):")
    if raw_data_csv_file:
        df = pd.read_csv(raw_data_csv_file)
    elif raw_data_csv_link:
        df = pd.read_csv(raw_data_csv_link)
    else:
        df = pd.read_csv(DEFAULT_CSV_URL)
     # Optionally save predictions to a CSV file covering the duration selected
        if st.button('Save Predictions'):
            # Provide a download link for the predictions CSV
            csv_data = predictions_df.to_csv(index=False).encode()
            b64 = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="stock_predictions.csv">Click here to download the predictions CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)

     
if __name__ == "__main__":
    main()
