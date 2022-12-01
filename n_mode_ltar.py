from multiprocessing.sharedctypes import Value
from statsmodels.tsa.api import VAR
import scipy.fft as sfft
import pywt
import numpy as np
import pandas as pd
import multiprocessing
import ctypes

def diff(data_tensor, interval = 1):
    # Data tensor needs to be N x m0 x m1 x .... x mn (where N is the amount of samples and n is the number of dimension of an observation)
    shape = data_tensor.shape
    new_shape = list(shape).copy()
    new_shape[0] -= interval
    new_shape = tuple(new_shape)

    diff = np.zeros(new_shape)
    for i in range(interval, shape[0]):
        diff[i - interval] = data_tensor[i] - data_tensor[i - interval]
    return diff

def invert_diff(diff_tensor, y0_tensor, interval = 1):    
    # Tensors needs to be N x m0 x m1 x .... x mn (where N is the amount of samples and n is the number of dimension of an observation)
    shape = diff_tensor.shape
    N = shape[0]
    result = diff_tensor.copy()

    for i in range(N):
        accum = 0
        for j in range(i // interval):
            accum += diff_tensor[i - (j+1)*interval]
        result[i] += y0_tensor[-interval + i % interval] + accum
    return result

def err(true_tensor, forecast_tensor):
    # Tensors needs to be N x m0 x m1 x .... x mn (where N is the amount of samples and n is the number of dimension of an observation)
    shape = true_tensor.shape
    if shape != forecast_tensor.shape:
        raise ValueError("True tensor and forecast tensor must have the same shape!")
    N = shape[0]

    errs = np.zeros((N,))
    for i in range(N):
        errs[i] = np.linalg.norm(true_tensor[i] - forecast_tensor[i]) / np.linalg.norm(true_tensor[i])
    return np.mean(errs)

class LTAR():
    def __init__(self):
        self.p = -1

    def __apply_trans(self, A, transformation, axis):            

        if transformation == "dwt":

            # Only allow even axis size
            if A.shape[axis] % 2 != 0:
                raise ValueError(f"{A.shape[axis]} is not a valid axis size for DWT. Only even sizes are allowed")

            cA,cD = pywt.dwt(A, "haar", axis=axis)
            result = np.append(cA, cD, axis=axis)
        elif transformation == "dct":
            result = sfft.dct(A, axis=axis)
        elif transformation == "fft":
            # Warning, may be funny behavior because I havent fully accounted for complex values
            result = np.fft.fft(A, axis=axis)
        else:
            raise ValueError(f"{transformation} is not a valid transformation")

        return result

    def __apply_inverse_trans(self, A, transformation, axis):

        if transformation == "dwt":
            cA,cD = np.split(A, 2, axis=axis)
            result = pywt.idwt(cA,cD, "haar", axis=axis)
        elif transformation == "dct":
            result = sfft.idct(A, axis=axis)
        elif transformation == "fft":
            result = np.fft.ifft(A, axis=axis)
            result = result.real
        else:
            raise ValueError(f"{transformation} is not a valid transformation")

        return result

    def recursive_fit_vars(self, X_sub, coef_sub, c_sub, curr_axis, p):

        # Base case
        if curr_axis == 2:

            # Train the VAR model
            tubes = X_sub[:,:,0] # n x l
            var = VAR(tubes)
            fit = var.fit(p)
        
            # Put them in their final coef tensor
            for k in range(p):
                coef_sub[k,:,:] = fit.coefs[k]
            c_sub[:,0] = fit.params[0,:]

            return # Dont need to return the changes since numpy slicing is done by reference

        # Fit all var models under this current axis
        for i in range(X_sub.shape[-1]):
            self.recursive_fit_vars(X_sub[..., i], coef_sub[..., i], c_sub[..., i], curr_axis - 1, p)

    def fit(self, X, p, transformation="dct"):

        # TENSORS ARE ASSUMED TO BE N x m0 x m1 x ... x mn
        self.tensor_shape = X.shape[1:]

        X = np.expand_dims(X, axis=2) # Expands to N x m0 x 1 x m1 x m2 x ... x mn

        if p < 1:
            raise ValueError(f"{p} is an invalid lag")
        self.p = p
        self.transformation = transformation

        # Applies the transformations
        i = len(X.shape) - 1
        X_trans = np.array(X)
        while i > 2:
            X_trans = self.__apply_trans(X_trans, transformation, i)
            i -= 1

        # Creates our parameter tensors
        coef_shape = [p, self.tensor_shape[0], self.tensor_shape[0]]
        c_shape = [self.tensor_shape[0], 1]
        for m_i in self.tensor_shape[1:]:
            coef_shape.append(m_i)
            c_shape.append(m_i)
        coef_shape = tuple(coef_shape)
        c_shape = tuple(c_shape)
        self.coefs = np.zeros(coef_shape) # Shape is p x m0 x m0 x m1 x m2 x ... x mn
        self.c = np.zeros(c_shape) # Shape is m0 x 1 x m1 x m2 x ... x mn

        # Fits the var models 
        self.recursive_fit_vars(X_trans, self.coefs, self.c, len(X.shape) - 1, p)

        # Applies the inverse transformation
        i = 3
        while i < len(X.shape):
            self.coefs = self.__apply_inverse_trans(self.coefs, transformation, i)
            self.c = self.__apply_inverse_trans(self.c, transformation, i-1)
            i += 1

    def rec_tprod(self, A_sub, B_sub, C_sub, curr_axis, c_shape):

        # Base case
        if curr_axis == 1:
            C_sub[:,:] = A_sub @ B_sub
            return # No need to return anything since python slicing is done by reference

        for i in range(c_shape[curr_axis]):
            self.rec_tprod(A_sub[..., i], B_sub[..., i], C_sub[..., i], curr_axis - 1, c_shape)


    def tprod(self, A, B, transformation):
        
        # A is assumed to be m0 x l x m1 x m2 x ... x mn
        # B is assumed to be l x m1 x m2 x ... x mn
        # Which results in A * B = C to be m0 x m1 x ... x mn

        # checks if the shapes are valid
        if A.shape[1]!=B.shape[0]:
            raise ValueError("A and B are invalid shapes!")
        for i in range(2, len(A.shape)):
            if A.shape[i] != B.shape[i]:
                raise ValueError("A and B are invalid shapes!")

        # Creates the new shape
        c_shape = [A.shape[0]]
        for m_i in B.shape[1:]:
            c_shape.append(m_i)
        c_shape = tuple(c_shape)

        i = 2
        Ahat = np.array(A)
        Bhat = np.array(B)
        while i < len(A.shape):
            Ahat = self.__apply_trans(Ahat, transformation, axis=i)
            Bhat = self.__apply_trans(Bhat, transformation, axis=i)
            i += 1

        C = np.zeros(c_shape)
        self.rec_tprod(Ahat, Bhat, C, len(A.shape) - 1, c_shape)

        i = len(A.shape) - 1
        while i >= 2:
            C = self.__apply_inverse_trans(C, transformation, axis=i)
            i -= 1
        return C

    def forecast(self, start, forecast_length):
        if len(start) != self.p:
            raise ValueError("Invalid start amount. Must equal to p...")

        start = np.expand_dims(start, axis=2)
        forecast_shape = [forecast_length+self.p, self.tensor_shape[0], 1]
        for m_i in self.tensor_shape[1:]:
            forecast_shape.append(m_i)
        forecast_shape = tuple(forecast_shape) # Shape is N_forecast + p x m0 x 1 x m1 x m2 x ... x mn

        forecast_tensor = np.zeros(forecast_shape)
        forecast_tensor[:self.p] = start
        for i in range(self.p,forecast_length+self.p):
            total = np.zeros(forecast_shape[1:])
            for j in range(self.p):
                total += self.tprod(self.coefs[j], forecast_tensor[i-j-1], self.transformation)
            forecast_tensor[i] = total + self.c
        return np.squeeze(forecast_tensor[self.p:], axis=2)