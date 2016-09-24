from WikilinksStatistics import *
import random
import nltk
from nltk.corpus import stopwords
import operator
import itertools
import math

class KnockoutModel:
    """
    This model takes a pairwise model that can train/predict on pairs of candidates for a wikilink
    and uses it to train/predict on a list candidates using a knockout method.
    The candidates are taken from a stats object
    """

    def __init__(self, pairwise_model, stats):
        """
        :param pairwise_model:  The pairwise model used to do prediction/training on a triplet
                                (wikilink,candidate1,candidate2)
        :param stats:           A statistics object used to get list of candidates
        """
        self._stats = stats
        self._pairwise_model = pairwise_model

    def predictRepeated(self, wikilink, candidates=None, repeats=20):
        if candidates is None and self._stats is None:
            #cant do nothin'
            return None

        if candidates is None:
            candidates = self._stats.getCandidatesForMention(wikilink["word"])
            candidates = {int(x): y for x, y in candidates.iteritems()}

        # do a knockout
        l = [candidate for candidate in candidates.keys()]

        ranking = {x:0.0 for x in l}

        if math.pow(2.0, len(l)) <= repeats:
            comb = itertools.permutations(l)
            for perm in comb:
                predicted = self._predict(wikilink, perm)
                if predicted is not None:
                    ranking[predicted] += 1.0
        else:
            for i in xrange(repeats):
                random.shuffle(l)
                predicted = self._predict(wikilink, l)
                if predicted is not None:
                    ranking[predicted] += 1.0

        m = max(ranking.iteritems(), key=operator.itemgetter(1))[0]
        mv = max(ranking.iteritems(), key=operator.itemgetter(1))[1]
        if m == 0:
            return None
        finals = {x: candidates[x] for x,y in ranking.items() if y == mv}
        final = max(finals.iteritems(), key=operator.itemgetter(1))[0]
        return final


    def predict(self, wikilink, candidates=None):
        if candidates is None and self._stats is None:
            #cant do nothin'
            return None

        if candidates is None:
            candidates = self._stats.getCandidatesForMention(wikilink["word"])
            candidates = {int(x): y for x, y in candidates.iteritems()}

        # do a knockout
        l = [candidate for candidate in candidates.keys()]
        random.shuffle(l)
        return self._predict(wikilink, l)

    def predict2(self, wikilink, candidates=None):
        if candidates is None and self._stats is None:
            #cant do nothin'
            return None

        if candidates is None:
            candidates = self._stats.getCandidatesForMention(wikilink["word"])
            candidates = {int(x): y for x, y in candidates.iteritems()}

        l = [candidate for candidate in candidates.keys()]
        if len(l) == 1:
            return l[0]

        ranking = {x:0.0 for x in l}

        for i in xrange(len(l) - 1):
            for j in xrange(i + 1, len(l)):
                a = self._pairwise_model.predict(wikilink, l[i], l[j])
                if a is not None:
                    ranking[a] += 1
                b = self._pairwise_model.predict(wikilink, l[j], l[i])
                if b is not None:
                    ranking[b] += 1

        m = max(ranking.iteritems(), key=operator.itemgetter(1))[0]
        mv = max(ranking.iteritems(), key=operator.itemgetter(1))[1]
        if m == 0:
            return None
        finals = {x: candidates[x] for x,y in ranking.items() if y == mv}
        final = max(finals.iteritems(), key=operator.itemgetter(1))[0]
        return final


    def _predict(self, wikilink, l):

        while len(l) > 1:
            # create a list of surviving candidates by comparing couples
            next_l = []

            for i in range(0, len(l) - 1, 2):
                a = self._pairwise_model.predict(wikilink, l[i], l[i+1])
                if a is not None:
                    next_l.append(a)

            if len(l) % 2 == 1:
                next_l.append(l[-1])
            l = next_l

        if len(l) == 0:
            return None
        return l[0]
