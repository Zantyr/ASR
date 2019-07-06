import keras
import keras.backend as K
from keras.layers import Lambda, Input
from keras.models import Model
import numpy as np

from fwk.stage_meta import Loss
from fwk.stage_selection_adapter import RandomSelectionAdapter

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
