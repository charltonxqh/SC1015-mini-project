#!/usr/bin/env python
# coding: utf-8

# # LSTM Analysis 
# 
# Long Short-Term Memory (LSTM) networks, a special kind of Recurrent Neural Network (RNN), have emerged as a powerful tool for predicting time series data due to their ability to capture long-term dependencies and patterns. Unlike traditional neural networks, LSTMs can remember information over extended periods, making them particularly suitable for applications where historical context significantly influences future outcomes. This capability is especially valuable in financial markets, where past stock prices and trends can provide insightful cues for future movements.
# 
# In this analysis, we leverage the strengths of LSTM networks to predict the stock price of the Vanguard S&P 500 ETF ($VOO)
# 
# 

# ### 0. Importing Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import seaborn as sns

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# ### 1. Data Preparation & Preprocessing
# 
# 

# In[2]:


voo_data = pd.read_csv('../datasets/VOO.csv')
df = pd.read_csv('../datasets/VOO.csv')
display(voo_data)


# In[3]:


# Convert the 'Date' column to datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Now, focus on the 'Close' column
df_close = df[['Close']]
df_close


# In[4]:


# Since mplfinance expects certain column names for the OHLC data, ensure your DataFrame columns are named appropriately
df = df[['Open', 'High', 'Low', 'Close']]

# Create the candlestick chart
mpf.plot(df, type='candle', style='charles', title='VOO Stock Price', ylabel='Price ($)')


# In[ ]:





# ### 2. Exploratory Data Analysis
# 

# In[5]:


#Summary Statistics
print(voo_data.describe())


# In[6]:


# Check for Missing Values
print(voo_data.isnull().sum())


# In[7]:


# Correlation Matrix
correlation_matrix = voo_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for VOO Variables')
plt.show()


# From this heatmap, we can conclude that the Open, High, Low, and Close values are highly correlated with each other for VOO, meaning they tend to move together during the trading period. However, Volume does not strongly correlate with price movement, indicating that for VOO, volume changes are not necessarily associated with large changes in price. This can be useful information for trading strategies that might, for example, use volume as an indicator independent of price movements.

# ---

# In[8]:


# Closing Price and Moving Averages on One Chart

plt.figure(figsize=(14, 7))

# Plotting the Closing Price
df['Close'].plot(label='Close Price')

# Short-term (50-day) and Long-term (200-day) Moving Averages
df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()
df['MA50'].plot(label='50-Day MA')
df['MA200'].plot(label='200-Day MA')

plt.title('VOO Close Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In the chart, the MA200 starts later than the MA50 (50-day moving average) and the closing price line. This happens because a 200-day moving average requires 200 days of data before the first value can be calculated. Therefore, the MA200 line will begin to appear on the chart only after the first 200 days.
# 
# The 200-day moving average is used to determine the long-term market trend and smooth out price fluctuations. It's slower to respond to price changes than the 50-day moving average because it considers a larger set of data, which is why it appears smoother and starts later in the chart.
# 
# The overall chart shows the closing price of VOO, along with the short-term (MA50) and long-term (MA200) trends. The MA50 line reacts more quickly to recent price changes, while the MA200 provides a more gradual trend line that reflects longer-term price movements. When the closing price dips below these averages, it could be seen as a bearish signal, and when it's above, it could be bullish. The intersection points where the closing price or MA50 crosses the MA200 can be of particular interest to traders looking for trend reversals.

# In[9]:


# Daily Returns and Volatility in a Single Chart
df['Daily Return'] = df['Close'].pct_change()

fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Return', color=color)
ax1.plot(df.index, df['Daily Return'], color=color, label='Daily Return')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Volatility (Rolling Std Dev of Daily Return)', color=color)  
ax2.plot(df.index, df['Daily Return'].rolling(window=30).std(), color=color, label='Rolling 30-Day Std Dev')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Daily Returns and Volatility')
plt.show()


# When the red line moves sharply away from the zero line, it indicates significant price changes from the previous day.
# Peaks and troughs in the blue line show periods of high and low volatility, respectively. A higher blue line indicates that the price of the asset was fluctuating more during that period, signaling higher risk. Conversely, a lower blue line indicates less fluctuation and lower risk.
# One can also look for patterns or correlations between the two lines. For instance, if large spikes in daily returns (red line) coincide with peaks in volatility (blue line), it suggests that higher returns are associated with higher risk.
# It's also noteworthy to see if periods of increased volatility lead or follow large changes in daily returns, which might inform a trading strategy or risk management approach.
# This kind of chart is particularly useful for traders and investors who wish to understand the risk-return profile of an asset over time and might be used to make decisions about timing entries and exits into the market.

# ---
# 

# ### 3. Training Set Preparation and Data Normalisation

# In[10]:


# Prepare training set
train_df = df_close

# Normalise data
scaler = MinMaxScaler(feature_range=(0,1))      
scaled_data = scaler.fit_transform(train_df['Close'].values.reshape(-1,1))

# Number of days to base prediction on:
N_PRED_DAYS = 50

x_train, y_train = [], []

for d in range(N_PRED_DAYS, len(scaled_data)):
    # Add previous days values to x_train
    x_train.append(scaled_data[d - N_PRED_DAYS: d, 0])
    # Add current day's value to y_train
    y_train.append(scaled_data[d, 0])

# Convert into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train, y_train


# ---

# ### 4. LSTM Model Building & Training

# In[11]:


def LSTM_model():
    
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))
    
    return model


# #### Training

# In[12]:


model = LSTM_model()
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')


# In[21]:


checkpointer = ModelCheckpoint(filepath='weights_best.weights.h5',  
                               verbose=2, 
                               save_best_only=True, 
                               save_weights_only=True,  
                               monitor='loss')

history = model.fit(x_train, 
                    y_train, 
                    epochs=35, 
                    batch_size=32,
                    callbacks=[checkpointer])


# ---

# ### 5. Predictions and Visualization

# In[31]:


# Load the test dataset
test_df = pd.read_csv('../datasets/VOO_test.csv')

# Convert the 'Date' column to datetime format for plotting purposes
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df.set_index('Date', inplace=True)

# Now, focus on the 'Close' column
test_df_close = test_df[['Close']].values


# In[35]:


model_inputs = df_close[len(train_df) - N_PRED_DAYS:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)


# In[36]:


x_test = []

for d in range(N_PRED_DAYS, len(model_inputs)):
    x_test.append(model_inputs[d - N_PRED_DAYS: d, 0])

# Convert to numpy array and reshape to 3D array with appropriate dimensions for LSTM model
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the prices 
predicted_prices = model.predict(x_test)
# Perform an inverse transform to obtain actual values
predicted_prices = scaler.inverse_transform(predicted_prices)


# ### 6. Evaluation & Observation

# In[ ]:





# ---
