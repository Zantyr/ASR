import numpy as np
import scipy as sp
import scipy.signal as sps

from fwk.stage_meta import Analytic, DType, ToDo


class Window(Analytic):
    """
    Divide recording into uniform overlapping windows
    Those can form a basis to different transforms
    Can apply windowing function
    """
    def __init__(self, size, hop, win_func=None):
        self.size = size
        self.hop = hop
        self.win_func = win_func  # if not None, initialize
        self.previous = None

    def output_dtype(self, input_dtype):
        return DType("Array", [1 + input_dtype.shape[0] // self.hop, self.size], np.float32)

    def _function(self, recording):
        windowed = np.zeros([1 + recording.shape[0] // self.hop, self.size], np.float32)
        for ix in range(windowed.shape[0]):
            slice = recording[ix * self.hop : ix * self.hop + self.size]
            if len(slice) != self.size:
                slice = np.pad(slice, (0, self.size - len(slice)), 'constant')
            if self.win_func is not None:
                slice = self.win_func * slice
            windowed[ix, :] = slice
        return windowed


class EqualLoudnessWeighting(Analytic):
    """
    IEC 61672:2003
    Based on: https://gist.github.com/endolith/148112
    """
    def __init__(self, kind):
        assert kind in ["A", "B", "C"]
        if kind == "A":
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 0.17
            numerator = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
            denominator = sp.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                           [1, 4*np.pi * f1, (2*np.pi * f1)**2])
            denominator = sp.polymul(sp.polymul(denominator, [1, 2*np.pi * f3]), [1, 2*np.pi * f2])
        if kind == "B":
            f1 = 20.598997
            f2 = 158.5
            f4 = 12194.217
            A1000 = 1.9997
            numerator = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0]
            denominator = sp.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                           [1, 4*np.pi * f1, (2*np.pi * f1)**2])
            denominator = sp.polymul(denominator, [1, 2*np.pi * f2])
        if kind == "C":
            f1 = 20.598997 
            f4 = 12194.217
            C1000 = 0.0619
            numerator = [(2*np.pi*f4)**2*(10**(C1000/20.0)),0,0]
            denominator = sp.polymul([1,4*np.pi*f4,(2*np.pi*f4)**2.0],[1,4*np.pi*f1,(2*np.pi*f1)**2])
        self.filter = sps.bilinear(numerator, denominator, 16000)

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return input_dtype
        
    def _function(self, recording):
        return sps.filtfilt(*self.filter, recording)


class PCENScaling(ToDo):
    """
    """


class AdaptiveGainAndCompressor(ToDo):
    """
    """
