from SimpleW2VPairwiseModel import SimpleW2VPairwiseModel
from BaselinePairwiseModel import *
from GuessPairwiseModel import *
from KnockoutModel import *
from WikilinksStatistics import *
from Word2vecLoader import *


class ModelTrainer:
    """
    This class generates pairwise training examples from the corpus: (wikilink, candidate1, candidate2, correct)
    and feeds them to a model's train method
    """

    def __init__(self, iter, stats, model):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._iter = iter
        self._model = model
        self._stats = stats

        self.n_samples = 0

    def train(self):
        self._model.startTraining()

        for wikilink in self._iter.wikilinks():
            actual = wikilink['wikiId']

            candidates = self._stats.getCandidatesForMention(wikilink["word"])
            if candidates is None or len(candidates<2):
                return None

            # TODO: randomly sample candidates according to their likelihood for the given mention
            for wrong in candidates:
                if wrong == actual:
                    continue

                if random.randrange(2) == 0:
                    self._model.train(wikilink, actual, wrong, actual)
                else:
                    self._model.train(wikilink, wrong, actual, actual)

        self._model.finilizeTraining()
