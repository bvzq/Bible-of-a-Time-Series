# Import of libraries
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Import of data
data = pd.read_csv('./data.csv',sep=';')
data['month'] = pd.to_datetime(data['month'], format='%Y%m')
data = data.set_index('month')

# Plot of the Time Series
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111)
ax.plot(data.available_card, 'blue', label= "available")
plt.title('available card')
plt.ylabel('available X100M€')
plt.xlabel('data')
ax.legend(loc=2)
plt.grid(True)
plt.show()

# STL decomposition
series = data.available_card
result = STL(series).fit()
chart = result.plot()
plt.show()

def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
            
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    
    
# Smooth by the previous 5 days
plot_moving_average(data.available_card, 10, plot_intervals=True)