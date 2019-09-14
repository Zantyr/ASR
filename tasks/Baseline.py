import fwk.acoustic as acoustic
import fwk.stage as stage

import keras

from core import AbstractModelTraining


class BaselineModelSTFT(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.LogPowerFourier(),  # Frequency analysis in cochlea
            # Early neural effects - not used
            # Low level features - not used
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])

    
class BaselineModelMFCC(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.LogPowerFourier(),  # Frequency analysis in cochlea
            # Early neural effects - mel filters
            stage.MelFilterbank(),
            stage.DCTCompression(first=24),
            # Low level features
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])
