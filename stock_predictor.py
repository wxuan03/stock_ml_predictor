import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf

ticker = 'GOOG'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
data.Close.plot(title=f"{ticker} Stock Price")
plt.show()

