import abc
import gammatone.gtgram
import hashlib
import keras
import keras.backend as K
import librosa
import numpy as np
import scipy as sp
import scipy.signal as sps

from keras.layers import Input, Lambda
from keras.models import Model
from syntax import Show

from fwk.stage_meta import SelectionAdapter, Stage, NetworkableMixin, Loss, Analytic, Neural, ToDo, DType
from fwk.stage_selection_adapter import RandomSelectionAdapter, SpeakerSelectionAdapter
from fwk.stage_loss import CTCLoss
from fwk.stage_transforms import PlainPowerFourier, LogPowerFourier, TrainableCQT, TrainableCZT, CZT, CQT, CommonFateTransform
from fwk.stage_preprocessing import Window, EqualLoudnessWeighting, PCENScaling, AdaptiveGainAndCompressor
from fwk.stage_filterbanks import TriangularERB, HarmonicTriangularERB, OverlappingHarmonicTriangularERB, RoEx, GammatoneFilterbank, MelFilterbank
from fwk.stage_time_domain import GammaChirp, TimeRoex, TrainableConvolve, CARFAC
from fwk.stage_neural import EarlyDNN, EarlyConv2D, EarlyConv1D, SparseDNN, AntimonotonyLayer, RNN, LaterConv1D, LaterDNN, LaterSparse1D, TimeWarpingRNN, TimeWarpingCNN, Core, CNN2D
from fwk.stage_misc import LogPower



class Normalization(Stage):
    @property
    def trainable(self):
        True
        
    def output_dtype(self, input_dtype):
        pass

    def bind(self, previous):
        if previous is None:
            pass
        else:
            raise NotImplementedError()
    
    def map(self, recording):
        pass
    
    
class AbstractWavelet(ToDo):
    def __init__(self):
        pass


class AbstractFilter(ToDo):
    """
    Will represent phase shifts in the cochlea
    """


class ExcitationTrace(ToDo):
    """
    Add max of current and exponential smoothing of past features at each band
    """


def phonemic_map(phones, activation='softmax'):
    inp = keras.layers.Input((None, 512))
    outp = keras.layers.Dense(phones + 1, activation=activation)(inp)
    return Neural(keras.models.Model(inp, outp))

    