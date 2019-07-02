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


class SelectionAdapter(metaclass=abc.ABCMeta):
    """
    This should be an ABC for RandomSelection Adapter
    TODO
    """


class RandomSelectionAdapter(SelectionAdapter):
    def __init__(self, valid_percentage=0.1, test_percentage=0.1):
        self._train = self._valid = self._test = self._hash = None
        self.initialized = False
        self._valid_percentage = valid_percentage
        self._test_percentage = test_percentage

    def initialize(self, dataset):
        number = len(dataset.rec_fnames)
        train_threshold = 1. - self._valid_percentage - self._test_percentage
        valid_threshold = 1. - self._test_percentage
        selection = np.random.random(number)
        self._train = selection < train_threshold
        self._valid = (train_threshold < selection) & (selection <= valid_threshold)
        self._test = selection >= valid_threshold
        self._hash = hashlib.sha512(selection.tobytes()).digest().hex()[:16]

    @property
    def selection_hash(self):
        return self._hash
    
    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test


class DType(Show):
    def __init__(self, cls, shape, dtype):
        self.cls = cls
        self.shape = shape
        self.dtype = dtype
        
    @classmethod
    def of(cls, what):
        return NotImplementedError("")


class Stage(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def trainable(self):
        pass

    @abc.abstractmethod
    def output_dtype(self, input_dtype):
        pass

    @abc.abstractmethod
    def bind(self, previous):
        pass
    
    @abc.abstractmethod
    def map(self, recording):
        pass
    
    
class Neural(Stage):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
    
    @property
    def trainable(self):
        return True
    
    def output_dtype(self, input_dtype):
        # assert whether inut_dtype matches the input
        return DType("Tensor", self.graph.output_shape, self.graph.dtype)

    def bind(self, previous):
        if previous is None:
            return self.new_network()
        else:
            return keras.models.Model(previous.inputs, self.graph(previous.outputs))

    def join(self, previous):
        return keras.models.Model(previous.inputs, self.graph(previous.outputs))
        
    def map(self, recording):
        pass
    
    def new_network(self, shape):
        if callable(shape):
            return self.graph(shape.shape[-1])
        return self.graph
        
class Analytic(Stage):
    @property
    def trainable(self):
        False

    def bind(self, previous):
        self.previous = previous
        return self

    def map(self, recording):
        if self.previous is None:
            return self._function(recording)
        return self._function(self.previous.map(recording))


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



class Loss:
    pass



class CTCLoss(Loss):
    def __init__(self, optimizer=None, use_noisy=False, selection_adapter=None):
        self.optimizer = optimizer if optimizer else keras.optimizers.Adam(clipnorm=1.)
        self.use_noisy = use_noisy
        self.selection_adapter = selection_adapter if selection_adapter else RandomSelectionAdapter()
        
    def compile(self, network):
        label_input = Input(shape = (None,))
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_lambda = Lambda(lambda args:K.ctc_batch_cost(*args), output_shape=(1,), name='ctc')([label_input, network.outputs[0], input_length, label_length])
        model = Model([network.inputs[0], label_input, input_length, label_length], [loss_lambda])
        model.compile(loss=(lambda y_true, y_pred: y_pred), optimizer=self.optimizer)
        return model

    @property
    def selection_hash(self):
        return self.selection_adapter.selection_hash
    
    @property
    def requirements(self):
        if self.use_noisy:
            return ["noisy", "transcripts"]
        return ["clean", "transcripts"]
    
    def fetch_train(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.train
        if self.use_noisy:
            return [[
                dataset.noisy[selection],
                dataset.transcriptions[selection],
                dataset.noisy_lens[selection],
                dataset.transcription_lens[selection]
            ], np.zeros(dataset.noisy[selection].shape[0])]
        return [[
            dataset.clean[selection],
            dataset.transcriptions[selection],
            dataset.clean_lens[selection],
            dataset.transcription_lens[selection]
        ], np.zeros(dataset.clean[selection].shape[0])]
    
    def fetch_valid(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.valid
        if self.use_noisy:
            return [[
                dataset.noisy[selection],
                dataset.transcriptions[selection],
                dataset.noisy_lens[selection],
                dataset.transcription_lens[selection]
            ], np.zeros(dataset.noisy[selection].shape[0])]
        return [[
            dataset.clean[selection],
            dataset.transcriptions[selection],
            dataset.clean_lens[selection],
            dataset.transcription_lens[selection]
        ], np.zeros(dataset.clean[selection].shape[0])]
    
    def fetch_test(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.test
        if self.use_noisy:
            return [[
                dataset.noisy[selection],
                dataset.transcriptions[selection],
                dataset.noisy_lens[selection],
                dataset.transcription_lens[selection]
            ], np.zeros(dataset.noisy[selection].shape[0])]
        return [[
            dataset.clean[selection],
            dataset.transcriptions[selection],
            dataset.clean_lens[selection],
            dataset.transcription_lens[selection]
        ], np.zeros(dataset.clean[selection].shape[0])]


class Window(Analytic):
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
    
    
class MelFilterbank(Analytic):
    def __init__(self, n_mels, fmin=20, fmax=8000, sr=16000):
        self.mel_basis = None
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        self.mel_basis = librosa.filters.mel(self.sr, (input_dtype.shape[1] - 1) * 2, self.n_mels, self.fmin, self.fmax).T
        return DType("Array", [input_dtype.shape[0], self.n_mels], np.float32)
    
    def _function(self, recording):
        return np.dot(recording, self.mel_basis)
    
    
class AbstractWavelet(Analytic):
    def __init__(self):
        pass


class AbstractFilter(Analytic):
    """
    Will represent phase shifts in the cochlea
    """


class GammatoneFilterbank(Analytic):
    
    _sums = {
        "mean": lambda x: np.mean(x, axis=0),
        "log-mean": lambda x: np.log(np.abs(np.mean(x, axis=0))),
        "log-max": lambda x: np.log(np.max(np.abs(x), axis=0)),
    }
    
    def __init__(self, n_mels=24, sr=16000, window=512, hop=128, sum="log-mean"):
        self.scale = gammatone.gtgram.centre_freqs(sr, n_mels, 20)
        self.filterbank = gammatone.gtgram.make_erb_filters(sr, self.scale)
        self.window = window
        self.hop = hop
        self.sum = self._sums[sum]
        self.n_mels = n_mels
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [int(np.ceil(1 + (input_dtype.shape[0] - self.window) / self.hop)), self.n_mels], np.float32)
        
    def _function(self, recording):
        gt = gammatone.gtgram.erb_filterbank(recording[:], self.filterbank).T
        length = int(np.ceil(1 + (gt.shape[0] - self.window) / self.hop))
        ret = np.zeros([length, gt.shape[1]], np.float32)
        for i in range(length):
            ret[i, :] = self.sum(gt[i * self.hop : i * self.hop + self.window, :])
        return ret


class ExcitationTrace(Analytic):
    """
    Add max of current and exponential smoothing of past features at each band
    """
    
    
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


class FreqFilter(Analytic):
    def __init__(self, n_filts=24, sr=16000, window=512, hop=128, sum="log-mean"):
        fbank = lambda freq, bounds: self._fbank(freq, bounds, sr, window)
        self.scale = self._scale(sr, n_filts, 20)
        self.scale = np.array(sorted(self.scale))
        self.widths = np.concatenate([[0], self.scale, [sr]])
        self.widths = [(self.widths[x], self.widths[x + 2]) for x in range(len(self.scale))]
        self.filter = np.array([fbank(self.scale[x], self.widths[x]) for x in range(n_filts)], np.float32).T
        self.sr = sr
        self.n_filts = n_filts

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], self.n_filts], np.float32)
    
    def _function(self, recording):
        return np.dot(recording, self.filter)
    

class ERBFilter(FreqFilter):
    """
    ERBFilter base class, filter function should be supplied.
    """
    _scale = lambda self, *args: gammatone.gtgram.centre_freqs(*args)

    
class TriangularERB(ERBFilter):
    def _fbank(_, freq, bounds, sr, window):
        def filt_fn(x):
            if x < bounds[0] or x > bounds[1]:
                return 0.
            if x >= freq:
                return 1. - (x - freq) / (bounds[1] - freq)
            if x < freq:
                return 1. - (freq - x) / (freq - bounds[0])
        filt_fn = np.vectorize(filt_fn)
        num_window = window // 2 + 1
        filt = filt_fn(np.arange(num_window) * sr / window)
        return filt / filt.sum()

    
class HarmonicTriangularERB(ERBFilter): # TO DO
    def _fbank(_, freq, bounds, sr, window):
        def filt_fn(x):
            if x < bounds[0] or x > bounds[1]:
                return 0.
            if x >= freq:
                return 1. - (x - freq) / (bounds[1] - freq)
            if x < freq:
                return 1. - (freq - x) / (freq - bounds[0])
        filt_fn = np.vectorize(filt_fn)
        num_window = window // 2 + 1
        filt = np.zeros(num_window, np.float32)
        for i in range(5):
            filt += filt_fn(np.arange(num_window) * sr / window / (i + 1)) / (i+1)
        return filt / filt.sum()
    

class OverlappingHarmonicTriangularERB(ERBFilter): # TO DO
    def _fbank(_, freq, bounds, sr, window):
        bounds = freq - bounds[0], freq + bounds[1]
        def filt_fn(x):
            if x < bounds[0] or x > bounds[1]:
                return 0.
            if x >= freq:
                return 1. - (x - freq) / (bounds[1] - freq)
            if x < freq:
                return 1. - (freq - x) / (freq - bounds[0])
        filt_fn = np.vectorize(filt_fn)
        num_window = window // 2 + 1
        filt = np.zeros(num_window, np.float32)
        for i in range(5):
            filt += filt_fn(np.arange(num_window) * sr / window / (i + 1)) / (i+1)
        return filt / filt.sum()

    
        

class PCENScaling:
    """
    """
    