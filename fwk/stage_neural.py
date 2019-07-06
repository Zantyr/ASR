"""
All later layers of the network
"""

import keras
from keras.models import Model

from fwk.stage_meta import ToDo, Neural


class EarlyDNN(ToDo):
    pass


class EarlyConv2D(ToDo):
    pass


class EarlyConv1D(ToDo):
    pass


class SparseDNN(ToDo):
    pass


class AntimonotonyLayer(ToDo):
    pass


class RNN(ToDo):
    pass


class LaterConv1D(ToDo):
    pass


class LaterDNN(ToDo):
    pass


class LaterSparse1D(ToDo):
    pass


class TimeWarpingRNN(ToDo):
    pass


class TimeWarpingCNN(ToDo):
    pass


class Core(Neural):
    def __init__(self, width=512, depth=7):
        self.width, self.depth = width, depth
        
    def new_network(self, dtype):
        first = inp = keras.layers.Input(shape=[None, dtype.shape[1]])
        for i in range(self.depth):
            inp = keras.layers.GRU(self.width, activation='linear', return_sequences=True)(inp)
            inp = keras.layers.LeakyReLU(0.01)(inp)
            inp = keras.layers.BatchNormalization()(inp)
        outp = inp
        mdl = Model(first, outp)
        return mdl

    def bind(self, other):
        pass
