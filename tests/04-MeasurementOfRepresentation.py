import sys
sys.path.append(".")

import fwk.dataset as dataset
import fwk.metricization as metricization
import fwk.stage as stage
import fwk.noise_gen as noise_gen

import keras
import numpy as np

dset = dataset.Dataset(noise_gen.Static())
dset.get_from("datasets/clarin-long/data")
dset.select_first(1000)

comparator = metricization.Metricization([
    stage.Window(512, 512),
    stage.LogPowerFourier(),
    stage.MelFilterbank(20),
])
comparator.add_metric(metricization.CosineMetric())
comparator.add_metric(metricization.EuclidMetric())

metrics = comparator.on_dataset(dset)
metrics.summary()