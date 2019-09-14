"""
All methods that transfomrm from time to frequency domain and vice-versa
"""

from fwk.stage_meta import ToDo, Analytic, DType, Neural
from fwk.stage_layers import CZT

import numpy as np


class PlainPowerFourier(Analytic):
    def __init__(self):
        pass
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], input_dtype.shape[1] // 2 + 1], np.float32)
    
    def _function(self, recording):
        shape = list(recording.shape)
        shape[1] = shape[1] // 2 + 1
        mapped = np.zeros(shape, np.float32)
        for i in range(shape[0]):
            mapped[i] = np.abs(np.fft.rfft(recording[i])) ** 2
        return mapped


class LogPowerFourier(Analytic):
    def __init__(self):
        pass
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], input_dtype.shape[1] // 2 + 1], np.float32)
    
    def _function(self, recording):
        shape = list(recording.shape)
        shape[1] = shape[1] // 2 + 1
        mapped = np.zeros(shape, np.float32)
        for i in range(shape[0]):
            mapped[i] = -np.log(np.abs(np.fft.rfft(recording[i])) ** 2 + 2e-12)
        return mapped
    

# How to implement this? Original CQT is hard to do...
class TrainableCQT(Neural):
    def __init__(self, real=True):
        self.real = real
        self.hop_length = 128
        self.sr = 16000
    
    def new_network(self, recording):
        cqt_frequencies = ...
        Q = ...
        __early_downsample = ...
        fft_basis, n_fft, _ = __cqt_filter_fft
        for i in range(n_octaves):
            pass
        pass
        if self.real:
            mapped = K.abs(mapped)
        return mapped


class TrainableCZT(ToDo):
    def __init__(self):
        pass
        
    def new_network(self, dtype):
        first = inp = keras.layers.Input(shape=[None] + dtype.shape[1:])
        outp = CZT()(conv_layer)
        mdl = Model(first, outp)
        return mdl


class CZT(Analytic):
    def __init__(self, win_size=512, hop=128, z=1.0, w=None, real=False):
        self.win_size = win_size
        self.hop = hop
        self.z = z
        self.w = w if w is not None else 1.0
        self.real = real

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", input_dtype.shape,  np.float32 if self.real else np.complex64)
    
    def _function(self, record):
        shape = list(recording.shape)
        mapped = np.zeros(shape, np.float32 if self.real else np.complex64)
        func = (lambda x: x) if self.real else np.abs
        for i in range(shape[0]):
            mapped[i] = func((self.z ** np.arange(self.win_size) * record[i]).outer(self.w ** np.arange(self.win_size)))
        return mapped


def spec2d(image, x_win, y_win, x_hop=1, y_hop=1):
    x_size = (image.shape[0] + 1 - x_win) // x_hop
    y_size = (image.shape[0] + 1 - y_win) // y_hop
    spec = np.zeros([x_size, y_size, x_win, y_win], np.complex64)
    for x in range(x_size):
        for y in range(y_size):
            spec[x, y, :, :] = np.fft.fft2(image[x * x_hop : x * x_hop + x_win, y * y_hop : y * y_hop + y_win])
    return spec


class CommonFateTransform(Analytic):
    def __init__(self, x_win, y_win):
        self.x_win = x_win
        self.y_win = y_win

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0] + 1 - self.x_win, input_dtype.shape[1] + 1 - self.y_win, self.x_win, self.y_win], np.float32)
    
    def _function(self, recording):
        shape = list(recording.shape)
        mapped = np.zeros(shape, np.float64)
        for i in range(shape[0]):
            mapped[i] = np.fft.fft(recording[i])
        return spec2d(mapped, self.x_win, self.y_win)


class CQT(Analytic):
    def __init__(self, real=True):
        self.real = real
        self.hop_length = 128
        self.sr = 16000
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], input_dtype.shape[1] // 2 + 1], np.float32)
    
    def _function(self, recording):
        mapped = librosa.cqt(recording, sr=self.sr, hop_length=self.hop_length)
        if self.real:
            mapped = np.abs(mapped)
        return mapped