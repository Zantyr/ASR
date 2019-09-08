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
            # Modelling cortical processes
            stage.Core(width=512, depth=3),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])
