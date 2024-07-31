import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima


st.set_page_config(page_title='Finance', layout='wide' )
st.markdown (
    "<h1>Stock Price</h2>",
    unsafe_allow_html=True
)



# User Input

ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")
start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('today'))
forecast_horizon = st.number_input('Enter forecast horizon(days)', min_value=1, value=30)



# Fetch the Finance Data

data = yf.download(ticker, start=start_date, end=end_date)


#pre process the data

data.reset_index(inplace=True)
data['Date']= pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
ts_data = data['Adj Close']


#Decompose time series
decomposition = seasonal_decompose(ts_data, model="multiplicative", period=12)
st.markdown("<h1>Seasonal Decomposition</h1>")
st.pyplot(decomposition.plot())

# TRAIN TEST SPLIT
monthly_data = ts_data.resample("M").mean()
train, test = train_test_split(monthly_data, test_size=0.2, shuffle=False)


#model Fiting
model = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
st.write("Fitting the model..........  DONE  ")



#Generate Forecoast
forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)


# Plot the original data, fitted values, and forecast
plt.figure(figsize=(12, 6))
plt.plot(train, label='Original Data')
plt.plot(forecast.index, forecast, label='Forecast', color='green')
plt.fill_between(forecast.index, 
                 conf_int[:, 0], 
                 conf_int[:, 1], 
                 color='k', alpha=.15)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Auto ARIMA Forecasting')
st.pyplot(plt)

st.write('Forecasted values:')
st.write(forecast)