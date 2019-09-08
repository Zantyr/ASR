import pynini


class LanguageModel:
    def __init__(self):
        self.vocab = None
        self.coefficients = [10]  # To Be Implemented

    @classmethod
    def load(self, path):
        pass

    def join_fst(self, acoustic_fst):
        return pynini.compose(acoustic_fst, self.language_fst)
    
    def sentence_hypotheses(self, acoustic_fst):
        sentence = self.join_fst(acoustic_fst)
        sentence = pynini.pdt_shortestpath(sentence)
        return [self.vocab[x] for x in sentence]

