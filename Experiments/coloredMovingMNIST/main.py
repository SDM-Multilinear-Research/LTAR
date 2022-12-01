import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pickle
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '../../') # Should point to the directory with the n_mode_ltar
from n_mode_ltar import LTAR, diff, invert_diff, err
from mlds import LMLDS

from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Conv3D

import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def post_process(forecast):
    forecast[forecast > 0.5] = 1
    forecast[forecast <= 0.5] = 0
    return forecast

data = pickle.load(open("coloredMovingMNIST.pkl", "rb"))
data = np.divide(data, 255)
(N, l, m, v) = data.shape
N = 2000
data = data[:N]
print("Shape =", data.shape)

def train_lmlds(train, test):
    train = train.reshape((len(train),l*m,v))
    N_test = test.shape[0]
    print("Fitting LMLDS...")
    lmlds = LMLDS(train)
    lmlds.fit()
    forecast = lmlds.forecast(N_test).reshape((len(test),l,m,v))
    return err(test, forecast)

def train_ltar(train, test):
    N_test = test.shape[0]
    print("Fitting LTAR...")
    ltar = LTAR()
    ltar_p = 2
    ltar.fit(train, ltar_p, "dct")
    forecast = post_process(ltar.forecast(train[-ltar_p:], N_test))
    return err(test, forecast)

def train_ltari(train, test):
    N_test = test.shape[0]
    print("Fitting LTARI...")
    ltari = LTAR()
    ltari_p = 2
    diff_train = diff(train)
    ltari.fit(diff_train, ltari_p)
    diff_forecast = ltari.forecast(diff_train[-ltari_p:], N_test)
    forecast = post_process(invert_diff(diff_forecast, train[-1:]))
    return err(test, forecast)

def train_lstar(train, test):
    N_test = test.shape[0]
    print("Fitting LSTAR...")
    lstar = LTAR()
    lstar_p = 2
    interval = 8
    diff_train = diff(train, interval)
    lstar.fit(diff_train, lstar_p)
    diff_forecast = lstar.forecast(diff_train[-lstar_p:], N_test)
    forecast = invert_diff(diff_forecast, train, interval)
    return err(test, forecast)

def train_lstm(train, test):
    N_test = test.shape[0]
    print("Fitting LSTM...")
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            start_ix = i - n_steps
            # check if we are beyond the sequence
            if start_ix >= 0:
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[start_ix:i], sequence[i]
                X.append(seq_x)
                y.append(seq_y)
        return np.asarray(X), np.asarray(y)
    n_features = l*m*v
    n_steps = 10
    train_X, train_y = split_sequence(train.reshape((train.shape[0], l*m*v)), n_steps)
    lstm = Sequential()
    lstm.add(Conv3D(2, 3, activation='relu',return_sequences=True, input_shape=(n_steps, n_features)))
    lstm.add(GRU(256, activation='relu',return_sequences=True))
    lstm.add(GRU(64, activation="relu"))
    lstm.add(Dense(n_features))
    lstm.add(Dense(n_features))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(train_X, train_y, epochs=10, verbose=1)
    forecast = np.zeros((N_test, l, m, v))
    forecast[:n_steps] = lstm.predict(train[-n_steps:].reshape((1, n_steps, n_features)), verbose=0).reshape((1, l, m, v))
    for i in range(n_steps,N_test):
        currX = forecast[i-n_steps:i].reshape((-1, n_steps, n_features))
        forecast[i] = post_process(lstm.predict(currX, verbose=0).reshape((l, m, v)))
    #forecast = lstm.predict(test_X, verbose=0).reshape((N_test, l, m, v))
    print(test.shape, forecast.shape)
    return err(test, forecast)

def run_train(train_func, train, test):
    start = time.time()
    error = train_func(train,test)
    end = time.time()
    return error, end-start

print("Generating Cross Validation Errors")

i = 1995
step = 1
ltar_errs = []
ltari_errs = []
lstar_errs = []
lstm_errs = []
lmlds_errs = []
ltar_time= []
ltari_time = []
lstar_time = []
lstm_time= []
lmlds_time = []
while i < N:

    # Split train test
    test_start = i 
    test_end = min(i+step, N)
    train = data[:test_start]
    test = data[test_start:test_end]

    print(i, test_start, test_end)

    # Train the models

    fc_err, runtime = run_train(train_ltar, train, test)
    ltar_errs.append(fc_err)
    ltar_time.append(runtime)
    fc_err, runtime = run_train(train_ltari, train, test)
    ltari_errs.append(fc_err)
    ltari_time.append(runtime)
    fc_err, runtime = run_train(train_lstar, train, test)
    lstar_errs.append(fc_err)
    lstar_time.append(runtime)
    # fc_err, runtime = run_train(train_lmlds, train, test)
    # lmlds_errs.append(fc_err)
    # lmlds_time.append(runtime)
    fc_err, runtime = run_train(train_lstm, train, test)
    lstm_errs.append(fc_err)
    lstm_time.append(runtime)

    print(ltar_errs[-1], ltari_errs[-1], lstar_errs[-1], lstm_errs[-1])
    print(ltar_time[-1], ltari_time[-1], lstar_time[-1], lstm_time[-1])

    i += step

err_df = pd.DataFrame(np.array([ltar_errs, ltari_errs, lstar_errs, lstm_errs]).T, columns=["LTAR", "LTARI", "LSTAR", "LSTM"])
err_df.to_csv("results/err.csv", index = False)
time_df = pd.DataFrame(np.array([ltar_time, ltari_time, lstar_time, lstm_time]).T, columns=["LTAR", "LTARI", "LSTAR", "LSTM"])
time_df.to_csv("results/time.csv", index = False)

print(err_df.describe())
print(time_df.describe())