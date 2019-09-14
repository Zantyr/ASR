"""
Things that do not fit elsewhere
"""

from fwk.stage_meta import ToDo, Analytic

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
