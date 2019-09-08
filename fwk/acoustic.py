import dill
import keras
import numpy as np
import os
import shutil
import tempfile
import zipfile
from fwk.stage import Neural, Analytic, Normalization, Loss, DType, Stage

_defaults = {}


class AcousticLoader:
    """
    Creates loadable item from object
    Any pickleability should be done here
    """
    
    def __init__(self, item):
        if isinstance(item, Stage):
            if hasattr(item, "serialize"):
                self.value = item.serialize()
            else:
                self.value = item.__class__
    
    def load(self):
        return self.value


class StopOnConvergence(keras.callbacks.Callback):
    def __init__(self, max_repetitions=10):
        super().__init__()
        self.max_repetitions = max_repetitions

    def on_train_begin(self, logs=None):
        self.repetitions = 0
        self.last_loss = np.inf

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None:
            if loss > self.last_loss:
                self.repetitions += 1
            else:
                self.last_loss = loss
                self.repetitions = 0
            if self.repetitions > self.max_repetitions:
                self.model.stop_training = True


class MappingGenerator:
    def __init__(self, stages):
        self.stages = stages
    
    def get_all(self, dataset):
        mapping, network, loss = None, None, None
        stages = list(reversed(self.stages))
        while stages:
            if isinstance(stages[-1], Neural) or isinstance(stages[-1], Loss):
                break
            mapping = stages.pop().bind(mapping)
        dset_dtype = dataset.generate_dtype(mapping)
        stages.reverse()
        for stage in stages:
            if isinstance(stage, Neural):
                if network is None:
                    network = stage.new_network(mapping.output_dtype(dset_dtype))
                else:
                    network = stage.join(network)
            elif isinstance(stage, Loss):
                train_network = stage.compile(network)
                loss = stage
                break
            else:
                raise TypeError("Incorrect subtype of Stage")
        else:
            if network is not None:
                raise RuntimeError("Network has not been compiled")
        return mapping, train_network, network, loss
                
    def get(self, dataset):
        return self.get_all(dataset)[0]
                
                
class AcousticModel:
    
    _separate_lists = ["callbacks", "stages"]
    
    def __init__(self, stages, name=None, symbol_map=None, callbacks=None):
        self.stages = stages
        self.symbol_map = symbol_map
        self.dataset_signature = None
        self.split_signature = None
        self.built = False
        self.config = None
        self.name = name if name else "blind"
        self.metrics = []
        self.statistics = {}
        self.callbacks = callbacks or [
            keras.callbacks.TerminateOnNaN(),
            StopOnConvergence(4)
        ]
    
    def add_metric(self, metric):
        self.metrics.append(metric)
    
    def build(self, dataset, **config):
        self.config = config
        mapping_generator = MappingGenerator(self.stages)
        mapping, train_network, network, loss = mapping_generator.get_all(dataset)
        dataset.generate(mapping, loss.requirements)
        self.dataset_signature = dataset.signature
        if network is not None:
            mc = keras.callbacks.ModelCheckpoint('models/{}-{}.h5'.format("", "{epoch:08d}"),
                                     save_weights_only=False, period=5)
            config = {
                "batch_size":32,
                "callbacks":[mc] + self.callbacks,
                "validation_data": loss.fetch_valid(dataset),
                "epochs": 250,
            }
            self.network = network
            train_network.summary()
            train_network.fit(*loss.fetch_train(dataset), **config)
        self.statistics["loss"] = train_network.evaluate(*loss.fetch_test(dataset))
        self.split_signature = loss.selection_hash
        for metric in self.metrics:
            self.statistics[metric.name] = metric.calculate(*loss.fetch_test(dataset))
        # predict and calculate - loss has to have "calculate"
        self.built = True
                        
    def predict(self, recording):
        pass
    
    def to_wfst(self, recording):
        pass
    
    @classmethod
    def load(self, path):
        tmpdname = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path) as f:
                f.extractall(tmpdname)
            with open(os.path.join(tmpdname, "root.dill"), "rb") as f:
                new_one = dill.load(f)
            for k, v in new_one.__dict__.items():
                if isinstance(v, str):
                    if v.startswith("dill://"):
                        fname = os.path.join(tmpdname, v.split("://")[1])
                        with open(fname, "rb") as f:
                            new_value = dill.load(f)                    
                        setattr(new_one, k, new_value)
                    elif v.startswith("keras://"):
                        fname = os.path.join(tmpdname, v.split("://")[1])
                        new_value = keras.models.load_model(fname, custom_objects=_defaults)
                        # TODO: Add decompression of additional features (like custom tf functions)
                        setattr(new_one, k, new_value)
            return new_one
        finally:
            shutil.rmtree(tmpdname)

    def save(self, path, format=False, save_full=True):
        if format:
            pass # TODO: change path somehow
        tmpdname = tempfile.mkdtemp()
        try:
            separate_objects = {}
            for k in dir(self):
                v = getattr(self, k)
                if isinstance(v, keras.models.Model):
                    fname = k + ".h5"
                    v.save(os.path.join(tmpdname, fname))
                    node_path = "keras://" + fname
                    separate_objects[k] = (node_path, v)
                elif k in self._separate_lists:
                    fname = k + ".bin"
                    self._save_dill_builder(v, os.path.join(tmpdname, fname))
                    node_path = "dill-builder://" + fname
                    separate_objects[k] = (node_path, v)
            try:
                for k, v in separate_objects.items():
                    setattr(self, k, v[0])
                with open(os.path.join(tmpdname, "root.dill"), "wb") as f:
                    dill.dump(self, f)
                with zipfile.ZipFile(path, 'w') as zipf:
                    for folder, _, files in os.walk(tmpdname):
                        for fname in files:
                            zipf.write(os.path.join(folder, fname), fname)
            finally:
                for k, v in separate_objects.items():
                    setattr(self, k, v[1])
        finally:
            shutil.rmtree(tmpdname)
        
    def __str__(self):
        if self.built:
            return "<Trained acoustic model with loss {}>".format(self.statistics["loss"])
        else:
            return "<Untrained acoustic model>"
        
    __repr__ = __str__

    def summary(self, show=True):
        if self.built:
            statstring = "\n    ".join(["{}: {}".format(k, v) for k, v in self.statistics.items()])
            docstring = ("--------\nTrained acoustic model named \"{}\"\nDataset signature: {}\n"
                         "Dataset train-valid-test selector signature: {}\nStatistics:\n"
                         "    {}\n--------").format(self.name, self.dataset_signature, self.split_signature, statstring)
        else:
            docstring = "Untrained acoustic model named \"{}\"".format(self.name)
        if show:
            print(docstring)
        else:
            return docstring
        
    def _save_dill_builder(self, item, path):
        print(item)
        pass
    
    def _summarize(self, item):
        if any([isinstance(item, x) for x in (tuple, list, set)]):
            return [_summarize(x) for x in item]
        elif any([isinstance(item, x) for x in (dict)]):
            return {k: _summarize(v) for k, v in item.items()}
        elif any([isinstance(item, x) for x in (str, int, float, type(None))]):
            return item
        else:
            return AcousticLoader(item)
        
    def _unsummarize(self, item):
        if isinstance(item, AcousticLoader):
            return item.load()
        elif hasattr(item, "__iter__"):
            if hasattr(item, "__items__"):
                return {k: _unsummarize(v) for k, v in item.items()}
            else:
                return [_unsummarize(k) for k in item]
        else:
            return item
