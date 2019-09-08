import os


class Type:
    """
    Class that composes type information and actual data
    Should be easily inheritable and constructable
    
    class MfccSeq(Type):
        ...
        
    wrapped = MfccSeq(data)
    
    Should have tree-traversing algorithm that checks castability
    Class should've class information registering all instances for traversal
    Should have methods for casting predefined
    """
    
    all_types = []
    # checks = []
    # castings = {}
    
    def __init__(self, data):
        Type.all_types.append(self)
        self.name = self.__class__.__name__
        if hasattr(self, "checks"):
            for check in self.checks:
                try:
                    assert check(data)
                except Exception as e:
                    e.message = "Typecheck failed for {}".format(self.name)
        self.data = data
        
    def get(self, data):
        return self.data


class Data:
    """
    Wekalike abstraction over datasets that contains metadata
    Works like a dictionary for pulling things
    """
    def __init__(self, **kwargs):
        self.data = {}
    
    def __getitem__(self, key):
        return self.data[key].get()
    
    def __setitem__(self, key, value):
        """
        This should access type,value tuple
        """
        pass
    
    def coalesce(self, key, new_type):
        pass


class Block:
    """
    This is a basic building block of Pipeline;
    
    Model is a subclass of Block
    """


class Pipeline:
    """
    Class implements serialization of models
    
    """
    
    def __init__(self, blocks=None, training=None, running=None, stateful=False):
        self.blocks = blocks if blocks is not None else blocks
        self.training = training if training is not None else training
        self.running = running if running is not None else running
        self.stateful = stateful
        
    @classmethod
    def load(cls, path):
        """
        Deserialize the model. Possible scenarios:
        - folder
        - zip archive
        """

    def save(self, path, archive=False):
        """
        Save model, to a folder or a zip
        """
        
    def train(self, *data):
        """
        Trains the model, can be called once
        TODO: multistage training
        """

    def evaluate(self, *data):
        """
        Returns results
        """

    def reset(self):
        """
        Resets a state of the module to a fresh one
        If stateless this does nothing
        """
