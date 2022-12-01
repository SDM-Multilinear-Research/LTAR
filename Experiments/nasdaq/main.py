import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '../../') # Should point to the directory with the n_mode_ltar
from n_mode_ltar import LTAR, diff, invert_diff, err
from mlds import LMLDS

from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Conv2D

import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

nasdaq = sio.loadmat('nasdaq100.mat')

N = 2175

tensor_shape = (N, 50, 4)

data = np.zeros(tensor_shape)
for i in range(tensor_shape[0]):
    data[i] = nasdaq['X'][i][0]

(N, l, m) = data.shape
print("Shape =", data.shape)

def train_lmlds(train, test):
    N_test = test.shape[0]
    print("Fitting LMLDS...")
    lmlds = LMLDS(train)
    lmlds.fit()
    forecast = lmlds.forecast(N_test)
    return err(test, forecast)

def train_ltar(train, test):
    N_test = test.shape[0]
    print("Fitting LTAR...")
    ltar = LTAR()
    ltar_p = 5
    ltar.fit(train, ltar_p)
    forecast = ltar.forecast(train[-ltar_p:],N_test)
    return err(test, forecast)

def train_ltari(train, test):
    N_test = test.shape[0]
    print("Fitting LTARI...")
    ltari = LTAR()
    ltari_p = 16
    diff_train = diff(train)
    ltari.fit(diff_train, ltari_p)
    diff_forecast = ltari.forecast(diff_train[-ltari_p:], N_test)
    forecast = invert_diff(diff_forecast, train)
    return err(test, forecast)

def train_lstar(train, test):
    N_test = test.shape[0]
    print("Fitting LSTAR...")
    lstar = LTAR()
    lstar_p = 5
    interval = 10
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
    n_features = l*m
    n_steps = 25
    train_X, train_y = split_sequence(train.reshape((train.shape[0], l*m)), n_steps)
    lstm = Sequential()
    lstm.add(Conv2D(2, 3, activation='relu',return_sequences=True, input_shape=(n_steps, n_features)))
    lstm.add(GRU(512, activation='relu',return_sequences=True))
    lstm.add(GRU(256, activation="relu"))
    lstm.add(Dense(128))
    lstm.add(Dense(n_features))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(train_X, train_y, epochs=300, verbose=1)
    forecast = np.zeros((N_test, l, m))
    forecast[:n_steps] = lstm.predict(train[-n_steps:].reshape((1, n_steps, n_features)), verbose=0).reshape((1, l, m))
    for i in range(n_steps,N_test):
        currX = forecast[i-n_steps:i].reshape((-1, n_steps, n_features))
        forecast[i] = lstm.predict(currX, verbose=0).reshape((l, m))
    #forecast = lstm.predict(test_X, verbose=0).reshape((N_test, l, m, v))
    print(test.shape, forecast.shape)
    return err(test, forecast)

print("Generating Cross Validation Errors")

def run_train(train_func, train, test):
    start = time.time()
    error = train_func(train,test)
    end = time.time()
    return error, end-start

i = 1200
step = 200
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
    fc_err, runtime = run_train(train_lmlds, train, test)
    lmlds_errs.append(fc_err)
    lmlds_time.append(runtime)
    fc_err, runtime = run_train(train_lstm, train, test)
    lstm_errs.append(fc_err)
    lstm_time.append(runtime)

    print(ltar_errs[-1], ltari_errs[-1], lstar_errs[-1], lmlds_errs[-1], lstm_errs[-1])
    print(ltar_time[-1], ltari_time[-1], lstar_time[-1], lmlds_time[-1], lstm_time[-1])

    i += step

err_df = pd.DataFrame(np.array([ltar_errs, ltari_errs, lstar_errs, lmlds_errs, lstm_errs]).T, columns=["LTAR", "LTARI", "LSTAR", "LMLDS", "LSTM"])
err_df.to_csv("results/err.csv", index = False)
time_df = pd.DataFrame(np.array([ltar_time, ltari_time, lstar_time, lmlds_time, lstm_time]).T, columns=["LTAR", "LTARI", "LSTAR", "LMLDS", "LSTM"])
time_df.to_csv("results/time.csv", index = False)

print(err_df.describe())
print(time_df.describe())

# print("Generating Samples of Time...")

# sample_errs = []
# start_i = 100
# i = start_i
# sample_step = 50
# test_size = 200
# sample_end = 500 #N - test_size
# while i < sample_end:

#     print(i)

#     train = data[:i]
#     test = data[i:i+test_size]

#     sample_errs.append([
#         np.mean(train_ltar(train,test)),
#         np.mean(train_ltari(train,test)),
#         np.mean(train_lstar(train,test)),
#         np.mean(train_lstm(train,test))
#     ])

#     i += sample_step

# sample_errs = np.array(sample_errs)
# steps = np.arange(start_i, sample_end, sample_step)
# plt.plot(steps, sample_errs[:,0])
# plt.plot(steps, sample_errs[:,1])
# plt.plot(steps, sample_errs[:,2])
# plt.plot(steps, sample_errs[:,3])
# plt.legend(["LTAR", "LTARI", "LSTAR", "LSTM"])
# plt.ylim(0, 1000)
# plt.savefig("taxi_samples_over_time.png")
# plt.close()

# for i in range(4):
#     with open(f"results/{names[i]}_samples.npy", "wb") as f:
#         np.save(f, sample_errs[:,i])
