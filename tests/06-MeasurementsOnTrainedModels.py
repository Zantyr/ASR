import sys
sys.path.append(".")

import fwk.acoustic as acoustic
import fwk.dataset as dataset
import fwk.metricization as metricization
import fwk.stage as stage
import fwk.noise_gen as noise_gen

import keras
import numpy as np

dset = dataset.Dataset(noise_gen.Static())
dset.get_from("datasets/clarin-long/data")
dset.select_first(100)

am = acoustic.AcousticModel([
    stage.Window(512, 512),
    stage.LogPowerFourier(),
    stage.MelFilterbank(20),
    stage.Core(width=512, depth=1),
    stage.phonemic_map(37),
    stage.CTCLoss()
])
am.build(dset)

comparator = metricization.RecordMetricization(am)
comparator.add_metric(metricization.PER())
metrics = comparator.on_dataset(dset)
metrics.summary()
