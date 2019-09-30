import dill
import keras
import numpy as np
import os
import pynini
import shutil
import tempfile
import time
import zipfile

import keras.backend as K
import tensorflow as tf

from fwk.stage import Neural, Analytic, Normalization, Loss, DType, Stage

_defaults = {"tf": tf}


class PeekGradients(keras.callbacks.Callback):
    """
    Used to debug NaNs
    """
    def on_batch_end(self, batch, logs=None):
        print("Layer weight statistics")
        for lyr in self.model.layers:
            print(lyr.name)
            for tensor in lyr.get_weights():
                print(tensor.mean(), "+-", tensor.std())
                



class AcousticLoader:
    """
    Creates loadable item from object
    Any pickleability should be done here
    """
    
    def __init__(self, item):
        self.builder = None
        if isinstance(item, Stage):
            if hasattr(item, "serialize"):
                self.value = item.serialize()
                if isinstance(self.value, tuple):
                    self.builder, self.value = self.value
            else:
                self.value = item.__class__
        elif any([isinstance(item, x) for x in (keras.callbacks.Callback,)]):
            self.value = item.__class__
        else:
            raise RuntimeError("No loader for {}".format(item))
    
    def load(self):
        if self.builder:
            return self.builder(self.value)
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


class Representation:
    def __init__(self, mapping, model):
        self.mapping = mapping
        self.model = model
    
    def map(self, recording):
        mapped = self.mapping.map(recording)
        if self.model:
            mapped = np.stack([mapped])
            mapped = self.model.predict(mapped)[0]
        return mapped


class MappingGenerator:
    def __init__(self, stages, representation_counter=None):
        self.stages = stages
        self.representation_mapping = None
        self.representation_counter = representation_counter
    
    def get_all(self, dataset):
        representation_cnt = 0
        mapping, train_network, network, loss = None, None, None, None
        stages = list(reversed(self.stages))
        while stages:
            if isinstance(stages[-1], Neural) or isinstance(stages[-1], Loss):
                break
            mapping = stages.pop().bind(mapping)
            representation_cnt += 1
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
            elif isinstance(stage, Analytic):
                if hasattr(stage, "to_network"):
                    network = stage.to_network(network)
                else:
                    raise TypeError("Analytic joint to network without to_network() method")
            else:
                raise TypeError("Incorrect subtype of Stage")
            representation_cnt += 1
            if representation_cnt == self.representation_counter:
                self.representation_mapping = keras.models.Model(network.inputs, network.outputs)
        else:
            if network is not None:
                raise RuntimeError("Network has not been compiled")
        return mapping, train_network, network, loss
                
    def get(self, dataset):
        return self.get_all(dataset)[0]
                
                
class AcousticModel:
    """
    representation_stage_count is used to produce representation mapping
    from trained neurals by taking N first stages during mapping;;
    
    Representation....
    """
    
    _separate_lists = ["callbacks", "stages"]
    
    def __init__(self, stages, name=None, symbol_map=None, callbacks=None, representation_stage_count=None, num_epochs=250):
        self.stages = stages
        self.symbol_map = symbol_map
        self.dataset_signature = None
        self.split_signature = None
        self.built = False
        self.config = None
        self.name = name if name else "blind"
        self.metrics = []
        self.statistics = {}
        self.complexity, self.building_time = None, None
        self.callbacks = callbacks or [
            keras.callbacks.TerminateOnNaN(),
            StopOnConvergence(4)
        ]
        self.representation_mapping = None
        self.num_epochs = num_epochs

    def add_metric(self, metric):
        self.metrics.append(metric)
    
    def build(self, dataset, **config):
        start_time = time.time()
        self.config = config
        mapping_generator = MappingGenerator(self.stages)
        mapping, train_network, network, loss = mapping_generator.get_all(dataset)
        dataset.generate(mapping, loss.requirements)
        print("Shape of the data: {}".format(dataset.clean.shape))
        self.dataset_signature = dataset.signature
        if network is not None:
            mc = keras.callbacks.ModelCheckpoint('models/{}-{}.h5'.format("", "{epoch:08d}"),
                                     save_weights_only=False, period=5)
            config = {
                "batch_size":32,
                "callbacks":[mc] + self.callbacks,
                "validation_data": loss.fetch_valid(dataset),
                "epochs": self.num_epochs,
            }
            self.network = network
            train_network.summary()
            train_network.fit(*loss.fetch_train(dataset), **config)
        self.statistics["loss"] = train_network.evaluate(*loss.fetch_test(dataset))
        self.split_signature = loss.selection_hash
        for metric in self.metrics:
            self.statistics[metric.name] = metric.calculate(*loss.fetch_test(dataset))
        # predict and calculate - loss has to have "calculate"
        self.building_time = time.time() - start_time
        self.mapping = mapping
        if not self.symbol_map:
            self.symbol_map = dataset.all_phones
        if mapping_generator.representation_mapping:
            self.representation_mapping = mapping_generator.representation_mapping
        self.complexity = None
        self.built = True

    def predict_raw(self, recording):
        if len(recording.shape) < 2:
            recording = np.stack([recording])
        dtype = self.mapping.output_dtype(DType("Array", recording.shape[1:], recording.dtype))
        mapped = np.zeros([recording.shape[0]] + dtype.shape, dtype=dtype.dtype)
        for i in range(recording.shape[0]):
            mapped[i] = self.mapping.map(recording[i])
        return self.network.predict(mapped)
        
    def predict(self, recording, literal=True, number_of_preds=1, beam_width=2500):
        predictions = self.predict_raw(recording)
        decoded = K.ctc_decode(predictions, [predictions.shape[1]] * predictions.shape[0], greedy=False, beam_width=beam_width, top_paths=number_of_preds)
        if literal:
            print(decoded)
            all_translations = []
            for recording in (decoded[0][x] for x in range(len(decoded[0]))):
                recording = recording.eval(session=K.get_session())
                print(recording)
                rec_translations = []
                for attempt in recording:
                    rec_translations.append([self.symbol_map[x] for x in attempt])
                all_translations.append(rec_translations)
            return all_translations
        else:
            return decoded
    
    def to_wfst(self, recording):
        phonemes = self.predict_raw(recording)
        EPSILON = 0
        fst = pynini.Fst()
        init = fst.add_state()
        fst.set_start(init)
        heads = [(init, EPSILON)]
        num_of_letters = phonemes.shape[2]
        time = phonemes.shape[1]
        letters = [x+1 for x in range(num_of_letters)]
        for time in range(time):
            states = [fst.add_state() for _ in letters]
            log_phonemes = -np.log(phonemes[0])
            for entering_state, head in heads:
                for letter, letter_state in zip(letters, states):
                    if letter == len(letters):
                        letter = 0
                    # letter_state = fst.add_state()
                    output_sign = head if head != letter else 0
                    weight = log_phonemes[time, letter]
                    fst.add_arc(entering_state, pynini.Arc(
                        letter, output_sign, weight, letter_state))
            heads = list(zip(states, letters))
        [fst.set_final(x[0]) for x in heads]
        if optimize:
            fst.optimize()
        return fst
    
    @classmethod
    def load(self, path):
        tmpdname = tempfile.mkdtemp()
        print(os.listdir(tmpdname))
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
                    elif v.startswith("dill-builder://"):
                        fname = os.path.join(tmpdname, v.split("://")[1])
                        with open(fname, "rb") as f:
                            new_value = dill.load(f)
                        new_value = self._unsummarize(new_value, tmpdname)
                        print(new_value)
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
                         "Dataset train-valid-test selector signature: {}\nTraining time: {}\n"
                         "Model complexity: {}\nStatistics:\n"
                         "    {}\n--------").format(self.name, self.dataset_signature, self.split_signature, self.building_time, self.complexity, statstring)
        else:
            docstring = "Untrained acoustic model named \"{}\"".format(self.name)
        if show:
            print(docstring)
        else:
            return docstring
        
    def _save_dill_builder(self, item, path):
        item = self._summarize(item, path)
        print(path)
        with open(path, "wb") as f:
            new_one = dill.dump(item, f)
    
    def _summarize(self, item, path):
        if any([isinstance(item, x) for x in (tuple, list, set)]):
            return [self._summarize(x, path) for x in item]
        elif any([isinstance(item, x) for x in (dict, )]):
            return {k: self._summarize(v, path) for k, v in item.items()}
        elif any([isinstance(item, x) for x in (str, int, float, type(None))]):
            return item
        else:
            return AcousticLoader(item)
        
    @classmethod
    def _unsummarize(self, item, tmpdname):
        if isinstance(item, AcousticLoader):
            return item.load()
        elif hasattr(item, "__iter__"):
            if hasattr(item, "__items__"):
                return {k: self._unsummarize(v, tmpdname) for k, v in item.items()}
            else:
                return [self._unsummarize(k, tmpdname) for k in item]
        else:
            return item

    def get_metrics(self, show=False):
        if show:
            return print(self.statistics.items())
        return self.statistics.items()
    
    def get_mapping(self):
        return Representation(self.mapping, self.representation_mapping)
