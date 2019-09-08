#!/usr/bin/python3

from lib.app import ResourceManager
from lib.training import Model, Trainer, AddLengths
from lib.core import Pipeline
from lib.loaders import seq_phones_loader

SAVE_PATH = ""

data = seq_phones_loader(
    "datasets/clarin-long/data/clarin-mfcc-rec-{}.npy",
    "datasets/clarin-long/data/clarin-mfcc-trans-{}.npy"
)

manager = ResourceManager()

from keras.models import Model
from keras.layers import LSTM, Conv1D, Dropout, LeakyReLU, Dense, Input, Lambda, TimeDistributed, Flatten, Conv2D, BatchNormalization, GRU
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.initializers import Orthogonal

from keras.backend import ctc_batch_cost, expand_dims
import keras.backend as K

def mk_model(max_label_length):
    feature_input = Input(shape = (None, NFEATS))
    layer = Lambda(lambda x: (x - MEAN) / STD)(feature_input)
    layer = Lambda(K.expand_dims)(layer)
    layer_1 = Conv2D(12, [5,1], activation='linear', strides=(2,1), kernel_initializer=Orthogonal(), padding='same')(layer)
    layer_2 = LeakyReLU(0.01)(layer_1)
    layer = BatchNormalization()(layer_2)
    layer_1 = Conv2D(16, [5,1], activation='linear', strides=(2,1), kernel_initializer=Orthogonal(), padding='same')(layer)
    layer_2 = LeakyReLU(0.01)(layer_1)
    layer = BatchNormalization()(layer_2)
    layer_1 = Conv2D(24, [7,1], activation='linear', kernel_initializer=Orthogonal(), padding='same')(layer)
    layer_2 = LeakyReLU(0.01)(layer_1)
    layer = BatchNormalization()(layer_2)
    layer = TimeDistributed(Flatten())(layer)
    layer_10 = LSTM(256, return_sequences = True, kernel_initializer=Orthogonal())(layer)
    layer_11 = LSTM(192, return_sequences = True, kernel_initializer=Orthogonal())(layer_10)
    layer_12 = LSTM(160, return_sequences = True, kernel_initializer=Orthogonal())(layer_11)
    layer_13 = LSTM(128, return_sequences = True, kernel_initializer=Orthogonal())(layer_12)
    layer_15 = LSTM(NPHONES + 1, return_sequences = True, activation = 'softmax')(layer_13)
    label_input = Input(shape = (max_label_length,))
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_lambda = Lambda(lambda args:ctc_batch_cost(*args), output_shape=(1,), name='ctc')([label_input, layer_15, input_length, label_length])
    model = Model([feature_input, label_input, input_length, label_length], [loss_lambda])
    model.summary()
    predictive = Model(feature_input, layer_15)
    return model, predictive

training, predictive = mk_model()
training = Model(training) # use mk_model somehow
trainer = Trainer(
    optimizer=Adam(0.0003, clip_norm=1.),
    loss={'ctc':lambda true, pred: pred},
    tensorboard=True
)
training.set_train(trainer)
predictive = Model(predictive)
add_lengths = AddLengths(...)
pipe = Pipeline(
    blocks={"training": training,
            "predictive": predictive,
            "add_lengths": add_lengths},
    training=["add_lengths", "training"],
    running=["predictive"]
)
pipe.train(data)
# TODO: add metrics and evaluation
pipe.save(SAVE_PATH)
