import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    .stDataFrame {
        border-radius: 5px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stPlot {
        border-radius: 5px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Stock Price Predictor")

# Sidebar for user input
with st.sidebar:
    st.header("🔍 Stock Selection")
    stock = st.text_input("Enter the Stock ID", "GOOG")
    st.write("Example: AAPL, MSFT, GOOG, AMZN")

    st.header("📅 Date Range")
    end_date = st.date_input("End Date", datetime.today())
    start_date = st.date_input("Start Date", datetime(end_date.year - 20, end_date.month, end_date.day))

    st.write(f"Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download stock data
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
google_data = yf.download(stock, start=start_date_str, end=end_date_str)

# Load the pre-trained model
try:
    model = load_model(r'Latest_stock_price_model.keras')
except Exception as e:
    st.error(f"⚠️ Model loading error: {e}")
    st.stop()

# Display stock data
st.subheader("📊 Stock Data")
st.dataframe(google_data.style.set_properties(**{'background-color': '#f5f5f5', 'color': 'black'}))

# Check if 'Close' column exists
if 'Close' not in google_data.columns:
    st.error("⚠️ The 'Close' column is missing in the downloaded data.")
    st.stop()

splitting_len = int(len(google_data) * 0.7)
x_test = google_data[['Close']][splitting_len:]

st.write(" x_test DataFrame (Sample):")
st.write(x_test.head())

# Plotting function with professional background
def plot_graph(figsize, values, full_data, extra_dataset=None, label_main="Moving Average", label_extra="Extra MA", color_main="#2ca02c", color_extra="#9467bd"):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f5f5f5')  # Light grey background
    ax.set_facecolor('#eaeaea')  # Slightly darker grey for the plot area
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    
    ax.plot(full_data.index, full_data['Close'], color="#2ca02c", label="Close Price")  # Green
    ax.plot(full_data.index, values, color=color_main, label=label_main)
    
    if extra_dataset is not None:
        ax.plot(full_data.index, extra_dataset, color=color_extra, label=label_extra)
    
    ax.legend()
    return fig

# Moving averages and plots
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()

st.subheader('📈 Close Price and 250-Day Moving Average')
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, label_main="250-Day MA", color_main="#d62728"))

st.subheader('📈 Close Price and 200-Day Moving Average')
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, label_main="200-Day MA", color_main="#d62728"))

st.subheader('📈 Close Price and 100-Day Moving Average')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, label_main="100-Day MA", color_main="#1f77b4"))

st.subheader('📊 Close Price with 100-Day and 250-Day Moving Averages')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, google_data['MA_for_250_days'], label_main="100-Day MA", label_extra="250-Day MA", color_main="#1f77b4", color_extra="#d62728"))

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(google_data[['Close']])
scaled_x_test = scaler.transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_x_test)):
    x_data.append(scaled_x_test[i - 100:i])
    y_data.append(scaled_x_test[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Model prediction
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create DataFrame for predictions
ploting_data = pd.DataFrame({
    'Original Test Data': inv_y_test.reshape(-1),
    'Predicted Test Data': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

st.subheader("📊 Original vs Predicted Stock Prices")
st.dataframe(ploting_data.style.set_properties(**{'background-color': '#f5f5f5', 'color': 'black'}))

# Final comparison plot
st.subheader('📈 Original vs Predicted Close Prices')
fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor('#f5f5f5')  # Light grey background
ax.set_facecolor('#eaeaea')  # Slightly darker grey for the plot area
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

ax.plot(google_data.index[:splitting_len + 100], google_data.Close[:splitting_len + 100], color="#1f77b4", label="Data - Not Used") #blue
ax.plot(ploting_data.index, ploting_data['Original Test Data'], color="#2ca02c", label="Original Test Data")  # Green
ax.plot(ploting_data.index, ploting_data['Predicted Test Data'], color="#d62728", label="Predicted Test Data")  # Red

ax.legend()
st.pyplot(fig)
