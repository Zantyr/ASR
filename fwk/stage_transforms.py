"""
All methods that transfomrm from time to frequency domain and vice-versa
"""

from fwk.stage_meta import ToDo, Analytic, DType

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
    

class TrainableDQT(ToDo):
    pass


class TrainableCZT(ToDo):
    pass


class CZT(ToDo):
    pass


class DQT(ToDo):
    pass


class CommonFateTransform(ToDo):
    pass
