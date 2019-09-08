import os

import fwk.acoustic as acoustic
import fwk.dataset as dataset


class Task(type):
    
    _instances = {}
    
    @classmethod
    def all(cls):
        return [cls._instances[x] for x in sorted(cls._instances.keys())]
    
    def __new__(self, name, bases, dct):
        new_dct = {"name": name, "implemented": True}
        new_dct.update(dct)
        item = super().__new__(self, name, bases, new_dct)
        self._instances[name] = item
        return item


class AbstractModelTraining(Task):
    def __new__(self, name, bases, dct):

        @classmethod
        def validate(self, cache):
            return os.path.exists(os.path.join(cache, "model.zip"))

        @classmethod
        def run(self, cache):
            try:
                if not os.path.exists(cache):
                    os.mkdir(cache)
            except:
                pass
            dset = dataset.Dataset()
            dset.get_from("datasets/clarin-long/data")
            dset.select_first(self.how_much)
            am = self.get_acoustic_model()
            am.name = name
            am.build(dset)
            am.summary()
            am.save(os.path.join(cache, "model.zip"), save_full=True)
            print("=" * 60)
            print("Task done!\n")

        @classmethod
        def summary(self, cache, show=False):
            am = acoustic.AcousticModel.load(os.path.join(cache, "model.zip"))
            return am.summary(show=show)
            
        new_dct = {"run": run, "validate": validate, "summary": summary, "how_much": 9000}
        new_dct.update(dct)
        return super().__new__(self, name, bases, new_dct)
