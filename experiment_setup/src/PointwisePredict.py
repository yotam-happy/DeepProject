from WikilinksStatistics import *
import random
import nltk
from nltk.corpus import stopwords
import operator
import itertools
import math
import operator

class PointwisePredict:
    """
    This model takes a pairwise model that can train/predict on pairs of candidates for a wikilink
    and uses it to train/predict on a list candidates using a knockout method.
    The candidates are taken from a stats object
    """

    def __init__(self, pointwise_model, stats):
        """
        :param pairwise_model:  The pairwise model used to do prediction/training on a triplet
                                (wikilink,candidate1,candidate2)
        :param stats:           A statistics object used to get list of candidates
        """
        self._stats = stats
        self._pointwise_model = pointwise_model

    def predict(self, wikilink, candidates=None):
        if candidates is None and self._stats is None:
            #cant do nothin'
            return None

        if candidates is None:
            candidates = self._stats.getCandidatesForMention(wikilink["word"])
            candidates = {int(x): y for x, y in candidates.iteritems()}

        if len(candidates) < 1:
            return None

        # do a knockout
        d = {candidate: self._predict(wikilink, candidate) for candidate in candidates.keys()}
        return max(d.iteritems(), key=operator.itemgetter(1))[0]

    def _predict(self, wikilink, candidate):
        a = self._pointwise_model.predict(wikilink, candidate)
        return a if a is not None else 0
