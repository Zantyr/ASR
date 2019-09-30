import os
import pynini


class LanguageModel:
    def __init__(self):
        self.vocab = None
        self.coefficients = [10]  # To Be Implemented

    @classmethod
    def load(self, path):
        pass
    
    def save(self):
        pass

    def join_fst(self, acoustic_fst):
        return pynini.compose(acoustic_fst, self.language_fst)
    
    def sentence_hypotheses(self, acoustic_fst):
        sentence = self.join_fst(acoustic_fst)
        sentence = pynini.pdt_shortestpath(sentence)
        return [self.vocab[x] for x in sentence]


    
    
    
class AcousticLanguage:
    def __init__(self, am, lm):
        self.am = am
        self.lm = lm
    
    def save(self, path):
        tmpdir
        self.am.save(os.path.join(tmpdir, "acoustic.zip"))
        self.lm.save(os.path.join(tmpdir, "language.zip"))

    @staticmethod
    def load(self, path):
        pass