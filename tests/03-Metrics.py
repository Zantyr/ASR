import sys
sys.path.append(".")

import fwk.acoustic as acoustic
import fwk.dataset as dataset
import fwk.stage as stage

import keras
import numpy as np

dset = dataset.Dataset()
dset.get_from("datasets/clarin-long/data")
dset.select_first(100)


def mk_model():
    inp = keras.layers.Input((None, 512))
    outp = keras.layers.Dense(38, activation='softmax')(inp)
    return keras.models.Model(inp, outp)

am = acoustic.AcousticModel([
    stage.Window(512, 512),
    stage.LogPowerFourier(),
    stage.MelFilterbank(20),
    stage.Core(width=512, depth=1),
    stage.Neural(mk_model()),
    stage.CTCLoss()
])

am.build(dset)
am.summary()
am.save("models/test-metrics.zip", format=False)
model = acoustic.AcousticModel.load("models/test-metrics.zip")
model.get_metrics(print=True)
noise = np.random.random([1, 16, 257]).astype(np.float32)
model.calculate_metric(noise)
