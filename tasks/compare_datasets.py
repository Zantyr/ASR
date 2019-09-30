"""
"""

from core import Task


class MetricMetaclass(Task):
    def __new__(self, name, bases, dct):
        @classmethod
        def validate(self, cache):
            return os.path.exists(os.path.join(cache, "metrics.bin"))

        @classmethod
        def run(self, cache):
            try:
                if not os.path.exists(cache):
                    os.mkdir(cache)
            except:
                pass
            self.representation = ...
            dataset = self.get_dataset()
            dataset.get_from("datasets/clarin-long/data")
            dataset.select_first(self.how_much)
            comparator = Metricization(self.representation)
            [comparator.add_metric(x) for x in [
                CosineMetric(),
                EuclidMetric()
            ]]
            metric = comparator.on_dataset(dataset)
            metric.save(os.path.join(cache, "metrics.bin"))
            print("=" * 60)
            print("Task done!\n")

        @classmethod
        def summary(self, cache, show=False):
            print(cache)
            am = Metrics.load(os.path.join(cache, "metrics.bin"))
            return am.summary(show=show)
            
        new_dct = {"run": run, "validate": validate, "summary": summary, "how_much": 9000}
        new_dct.update(dct)
        return super().__new__(self, name, bases, new_dct)

    
    
class PostModelMetricMetaclass(Task):
    def __new__(self, name, bases, dct):
        @classmethod
        def validate(self, cache):
            return os.path.exists(os.path.join(cache, "metrics.bin"))

        @classmethod
        def run(self, cache):
            try:
                if not os.path.exists(cache):
                    os.mkdir(cache)
            except:
                pass
            self.representation = ...
            dataset = self.get_dataset()
            dataset.get_from("datasets/clarin-long/data")
            dataset.select_first(self.how_much)
            comparator = Metricization(self.representation)
            [comparator.add_metric(x) for x in [
                CosineMetric(),
                EuclidMetric()
            ]]
            metric = comparator.on_dataset(dataset)
            metric.save(os.path.join(cache, "metrics.bin"))
            print("=" * 60)
            print("Task done!\n")

        @classmethod
        def summary(self, cache, show=False):
            print(cache)
            am = Metrics.load(os.path.join(cache, "metrics.bin"))
            return am.summary(show=show)
            
        new_dct = {"run": run, "validate": validate, "summary": summary, "how_much": 9000}
        new_dct.update(dct)
        return super().__new__(self, name, bases, new_dct)
    
"""    
class MetricForMP3(metaclass=MetricMetaclass):
    def get_dataset():
        return Dataset(noise_gen=CodecSox("mp3-lq"))


class MetricForGSM(metaclass=MetricMetaclass):
    def get_dataset():
        return Dataset(noise_gen=CodecSox("gsm"))


class MetricForAMR(metaclass=MetricMetaclass):
    def get_dataset():
        return Dataset(noise_gen=CodecSox("amr-nb"))
    

class MetricForWhiteNoise(metaclass=MetricMetaclass):
    def get_dataset():
        return Dataset(noise_gen=Static())
"""
"""
class MetricForRandomImpulse(metaclass=MetricMetaclass):
    def get_dataset():
        return Dataset(noise_gen=RandomImpulseResponse())
"""
