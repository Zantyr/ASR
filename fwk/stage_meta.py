import abc

import keras
from syntax import Show


class Watcher(type):
    """
    Based on https://stackoverflow.com/questions/18126552
    """

    count = 0
    
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            Watcher.count += 1
            print("Yet another class to be finished: " + name)
        super(Watcher, cls).__init__(name, bases, clsdict)


class ToDo(metaclass=Watcher):
    """
    This class will print that something is to be done
    """
    
    @staticmethod
    def status():
        print("Classes to be finished: {}".format(Watcher.count))


class SelectionAdapter(metaclass=abc.ABCMeta):
    """
    SelectionAdapter is a class for division between train, valid and test datasets
    The division may or may not take into account the speakers or other circumstances
    """
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

    
class Stage(metaclass=abc.ABCMeta):
    """
    """

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

    
class NetworkableMixin(metaclass=abc.ABCMeta):
    """
    You can add this Analytic to network
    """


class Loss(Stage, metaclass=abc.ABCMeta):
    """
    Loss is a strange Stage, as it does not really implement most of the methods
    TODO: To rework
    """    
    
    @property
    def trainable(self):
        return False

    def output_dtype(self, input_dtype):
        return None

    def bind(self, previous):
        return None
    
    def map(self, recording):
        return None

    @property
    @abc.abstractmethod    
    def selection_hash(self):
        pass
    
    @property
    @abc.abstractmethod
    def requirements(self):
        if self.use_noisy:
            return ["noisy", "transcripts"]
        return ["clean", "transcripts"]

    @abc.abstractmethod
    def fetch_train(self, dataset):
        pass
        
    @abc.abstractmethod
    def fetch_valid(self, dataset):
        pass
    
    @abc.abstractmethod
    def fetch_test(self, dataset):
        pass

    
class DType(Show):
    def __init__(self, cls, shape, dtype):
        self.cls = cls
        self.shape = shape
        self.dtype = dtype
        
    @classmethod
    def of(cls, what):
        return NotImplementedError("")
    
    
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
