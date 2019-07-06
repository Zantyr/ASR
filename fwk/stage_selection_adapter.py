import hashlib
import numpy as np

from fwk.stage_meta import SelectionAdapter, ToDo


class RandomSelectionAdapter(SelectionAdapter):
    """
    Divide recordings fully randomly. 
    This is generally the default selection adapter
    """
        
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

    
class SpeakerSelectionAdapter(ToDo):
    """
    Inherits: SelectionAdapter
    
    This should take into account speaker information in the dataset
    """
