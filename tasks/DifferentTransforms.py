from tasks.core import AbstractModelTraining


class LogPowerDQT(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.DQT(),  # Frequency analysis in cochlea
            stage.LogPower(),
            # Early neural effects - not used
            # Low level features
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])
    

class LogPowerDQTFilters(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.DQT(),  # Frequency analysis in cochlea
            stage.LogPower(),
            # Early neural effects - not used
            stage.TriangularERB(),
            # Low level features
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])

    
class TrainableDQTExperiment(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.NeuralDQT(),  # Frequency analysis in cochlea
            stage.LogPower(),
            # Early neural effects - not used
            # Low level features - not used
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])

    
class TrainableCZTExperiment(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.NeuralCZT(),  # Frequency analysis in cochlea
            stage.LogPower(),
            # Early neural effects - not used
            # Low level features - not used
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])


class CommonFateTransformTraining(metaclass=AbstractModelTraining):
    
    @classmethod
    def get_acoustic_model(self):
        return acoustic.AcousticModel([
            # Transmission model - not used
            stage.Window(512, 512),
            stage.CommonFateTransform(),  # Frequency analysis in cochlea
            stage.LogPower(),
            # Early neural effects - not used
            # Low level features - not used
            stage.CNN2D(channels=16, filter_size=5, depth=2),
            # Modelling cortical processes
            stage.RNN(width=512, depth=2),
            # Time warping
            stage.phonemic_map(37),
            stage.CTCLoss()
        ])
