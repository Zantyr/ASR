"""
Operations in time domain
"""

import keras

from fwk.stage_meta import ToDo, Neural


class GammaChirp(ToDo):
    def __init__(self):
        pass


class TimeRoex(ToDo):
    pass


class TrainableConvolve(Neural):
    def __init__(self, n_channels=48, winsize=512, stride=128, wavelet_constraint=True):
        self.n_channels = n_channels
        self.winsize = winsize
        self.stride = stride
        self.wavelet_constraint = wavelet_constraint
        
    def new_network(self, dtype):
        first = inp = keras.layers.Input(shape=[None] + self.n_channels)
        if len(dtype.shape) > 2:
            first = keras.layers.Flatten()(first)
        if self.wavelet_constraint:
            constraint = lambda x: (x - K.mean(x)) / K.std(x)
        else:
            constraint = None
        outp = keras.layers.Conv1D(self.n_channels, self.winsize, strides=self.stride, kernel_constraint=constraint, bias=(not self.wavelet_constraint))
        mdl = Model(first, outp)
        return mdl


class TrainableWavelet(ToDo):
    """
    This would require an implementation of custom layer
    """


class CARFAC(ToDo):
    pass
