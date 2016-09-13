from WikilinksStatistics import *
from Word2vecLoader import *


class ModelTrainer:
    """
    This class generates pairwise training examples from the corpus: (wikilink, candidate1, candidate2, correct)
    and feeds them to a model's train method
    """

    def __init__(self, iter, stats, model, epochs = 10,
                 wordInclude = None, wordExclude=None, senseFilter = None):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._neg_sample = 1
        self._iter = iter
        self._model = model
        self._stats = stats
        self._epochs = epochs
        self.wordInclude = wordInclude
        self.wordExclude = wordExclude
        self.senseFilter = {int(x) for x in senseFilter} if senseFilter is not None else None

        #setup all-sense negative-sampling
        self._all_senses = [int(x) for x in self._stats.conceptCounts.keys()]
        self._neg_sample_all_senses_prob = 0.00
        self._neg_sample_seenWith_prob = 0.1

        self._t = 0
        self._x1 = 0
        self._x2 = 0
        self._x3 = 0

    def getSenseNegSample(self):
        return self._all_senses[np.random.randint(len(self._all_senses))]

    def train(self):
        print "start training..."
        self._model.startTraining()

        for epoch in xrange(self._epochs):
            print "training epoch ", epoch

            for wikilink in self._iter.wikilinks():
                if self.wordExclude is not None and wikilink["word"] in self.wordExclude:
                    continue
                if self.wordInclude is not None and wikilink["word"] not in self.wordInclude:
                    continue

                actual = wikilink['wikiId']
                if self.senseFilter is not None and actual in self.senseFilter:
                    continue

                candidates = self._stats.getCandidatesForMention(wikilink["word"])
                if self.senseFilter is not None:
                    candidates = {x:y for x,y in candidates.iteritems() if x not in self.senseFilter}

                if len(candidates) == 0:
                    continue

                # get id vector
                ids = [candidate for candidate in candidates.items() if int(candidate[0]) != actual]

                for k in xrange(self._neg_sample):

                    # do negative sampling (get a negative sample)
                    r = np.random.rand()
                    if r < self._neg_sample_all_senses_prob:
                        # get negative sample from all possible senses
                        neg_candidates = self._all_senses
                        self._x1 += 1
                    elif r < self._neg_sample_all_senses_prob + self._neg_sample_seenWith_prob:
                        # get negative sample from senses seen with the correct one
                        neg_candidates = self._stats.getCandidatesSeenWith(actual).keys()
                        if len(neg_candidates) < 1:
                            continue
                        self._x2 += 1
                    else:
                        # get negative sample from senses seen for the current mention
                        neg_candidates = ids
                        if len(neg_candidates) < 1:
                            continue
                        self._x3 += 1
                    self._t += 1
                    #if self._t % 1000 == 0:
                    #    print "1: ", float(self._x1) / self._t, "2: ", float(self._x2) / self._t, "3: ", float(self._x3) / self._t
                    wrong = neg_candidates[np.random.randint(len(neg_candidates))]

                    # train on both sides so we get a symmetric model
                    if random.randrange(2) == 0:
                        self._model.train(wikilink, actual, wrong, actual)
                    else:
                        self._model.train(wikilink, wrong, actual, actual)

        self._model.finilizeTraining()
        print "done training."
