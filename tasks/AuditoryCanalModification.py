import fwk.acoustic as acoustic
import fwk.stage as stage

import keras

from core import AbstractModelTraining
    
    
class BaselineModelPCEN(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model
            stage.PCENScaling(),
            # Frequency analysis in cochlea
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            # Early neural effects - not used
            # Low level features - not used
            # Modelling cortical processes
            stage.RNN(width=512, depth=3),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])

    
class BaselineModelAdaptive(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model
            stage.AdaptiveGainAndCompressor(),
            # Frequency analysis in cochlea
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            # Early neural effects - not used
            # Low level features - not used
            # Modelling cortical processes
            stage.RNN(width=512, depth=3),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])


class BaselineModelEqualA(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model
            stage.EqualLoudnessWeighting("A"),
            # Frequency analysis in cochlea
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            # Early neural effects - not used
            # Low level features - not used
            # Modelling cortical processes
            stage.RNN(width=512, depth=3),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])


class BaselineModelEqualB(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model
            stage.EqualLoudnessWeighting("A"),
            # Frequency analysis in cochlea
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            # Early neural effects - not used
            # Low level features - not used
            # Modelling cortical processes
            stage.RNN(width=512, depth=3),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])


class BaselineModelEqualC(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model
            stage.EqualLoudnessWeighting("C"),
            # Frequency analysis in cochlea
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            # Early neural effects - not used
            # Low level features - not used
            # Modelling cortical processes
            stage.RNN(width=512, depth=3),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])
