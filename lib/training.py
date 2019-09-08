import keras
from lib.core import Block

class Model(Block):
    """
    Wraps a model (or transformer, does not matter)
    This specifies an interface for pipelining
    
    Any kind of transform may be here, a la sklearn transforms...
    
    This should have defined metadata for inputs and outputs
    This should validate the data
    This should accept train, validate, run routines
    QUESTION: should I wrap them? ANSWER: yes
    
    Should accept throttling
    """
    def load():
        pass
    
    def save():
        pass

    def train():
        pass
    
    def run():
        pass
    
    def set_train():
        pass
    
    def set_run():
        pass



class Trainer:
    """
    Class defining the training process
    This should be addable as a _train callback to Model
    
    Training requirements:
    - caching of models
    - logging to file
    - monitorable state - callbacks and such

    mc = keras.callbacks.ModelCheckpoint('models/mfcc-ctc-{epoch:08d}.h5', 
                                         save_weights_only=False, period=5)

    Class should define tensorboard service...
    """
    
    def __call__(self, model, *data):
        """
        Should build dynamically
        """



class AddLengths:
    """
    Class that gets lengths of two predefines sequences in the dataset
    """

    def run(self, dataset):
        dataset = dataset.copy() # should be shallow copy, only for references
        dataset.coalesce(self.input_sequence, self.input_seq_type)
        dataset.coalesce(self.output_sequence, self.output_seq_type)
        X = dataset[self.input_sequence]
        Y = dataset[self.output_sequence]
        data = X, Y
        NPHONES = Y.max()
        NFEATS = data[0].shape[2]
        X_lens = np.array([try_else(
                (lambda:np.where((x).mean(1) == (x).std(1))[0][0]),
                X.shape[1])
            for x in X])
        X_lens = np.ceil(X_lens / 4.0)
        Y_lens = np.array([np.where(x == NPHONES)[0] for x in data[1]])
        Y_lens = np.array([x[0] if len(x) else 0 for x in Y_lens])
        dataset[self.input_sequence] = X[np.where(Y_lens)]
        dataset[self.output_sequence] = Y[np.where(Y_lens)]
        dataset[self.input_sequence + "_lens"] = X_lens
        dataset[self.output_sequence + "_lens"] = Y_lens


