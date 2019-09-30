"""
Things that do not fit elsewhere
"""

from fwk.stage_meta import ToDo, Analytic, DType
from functools import reduce

import numpy as np


class Pointwise(ToDo):
    """
    Squash and so on
    """

    
class LogPower(Analytic):
    def __init__(self, negative=True):
        self.negative = negative
    
    def output_dtype(self, input_dtype):
        return input_dtype
    
    def _function(self, recording):
        return (-1 if self.negative else 1) * np.log(np.abs(recording))


class ConcatFeatures(Analytic):
    def __init__(self, feature_transforms, max_fit = 10):
        self.feature_transforms = feature_transforms
        self.max_fit = max_fit
        
    def bind(self, prev):
        self.previous = prev
        [reduce((lambda former, x: x.bind(former)), transform, prev)
            if isinstance(transform, list) else
            transform.bind(prev)
            for transform in self.feature_transforms]
        return self
    
    def output_dtype(self, input_dtype):
        dtypes = [
            reduce((lambda former, x: x.output_dtype(former)), transform, input_dtype)
            if isinstance(transform, list) else
            transform.output_dtype(input_dtype)
            for transform in self.feature_transforms]
        shape = sum([dtype.shape[-1] for dtype in dtypes])
        shape = dtypes[0].shape[:-1] + [shape]
        print(dtypes, shape, input_dtype)
        return DType("Array", shape, np.float32)
    
    def _function(self, recording):
        transforms = [
            reduce(lambda former, x: x._function(former), transform, recording)
            if isinstance(transform, list) else
            transform._function(recording)
            for transform in self.feature_transforms]
        times = np.array([x.shape[0] for x in transforms])
        times -= times.min()
        if times.max() < self.max_fit:
            max_time = np.array([x.shape[0] for x in transforms]).max()
            transforms = [
                np.pad(x, tuple([(0, max_time - x.shape[0])] + [
                    (0, 0) for dim in x.shape[1:]
                ]), 'constant') for x in transforms
            ]
        transforms = np.concatenate(transforms, axis=(len(transforms[0].shape) - 1))
        return transforms
