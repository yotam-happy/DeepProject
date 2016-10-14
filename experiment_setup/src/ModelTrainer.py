from WikilinksStatistics import *
from Word2vecLoader import *
from bisect import bisect

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

        # setup all-sense negative-sampling (define cumulative probability function)
        # -- some ppl say that for long lists it is better to have small probs first due to precision issues
        senses = [(int(x), int(y)) for x, y in self._stats.conceptCounts.items()]
        senses = sorted(senses, key=lambda tup: tup[1])
        self._all_senses, self._all_senses_cpf = [e[0] for e in senses], [e[1] for e in senses]
        self._all_senses_cpf_total = 0
        for i in xrange(len(self._all_senses_cpf)):
            self._all_senses_cpf_total += self._all_senses_cpf[i]
            self._all_senses_cpf[i] = self._all_senses_cpf_total

        self._neg_sample_uniform = True
        self._neg_sample_all_senses_prob = 0.0
        self._neg_sample_seenWith_prob = 0.0

        self._t = 0
        self._x1 = 0
        self._x2 = 0
        self._x3 = 0

    def getSenseNegSample(self):
        if (self._neg_sample_uniform):
            return self._all_senses[np.random.randint(len(self._all_senses))]
        x = np.random.randint(self._all_senses_cpf_total)
        i = bisect(self._all_senses_cpf, x)
        return self._all_senses[i]

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
                        wrong = self.getSenseNegSample()
                        self._x1 += 1
                    elif r < self._neg_sample_all_senses_prob + self._neg_sample_seenWith_prob:
                        # get negative sample from senses seen with the correct one
                        neg_candidates = self._stats.getCandidatesSeenWith(actual).keys()
                        if len(neg_candidates) < 1:
                            continue
                        wrong = neg_candidates[np.random.randint(len(neg_candidates))]
                        self._x2 += 1
                    else:
                        # get negative sample from senses seen for the current mention
                        neg_candidates = ids
                        if len(neg_candidates) < 1:
                            continue
                        wrong = neg_candidates[np.random.randint(len(neg_candidates))][0]
                        self._x3 += 1
                    self._t += 1

                    # train on both sides so we get a symmetric model
                    if random.randrange(2) == 0:
                        self._model.train(wikilink, actual, wrong, actual)
                    else:
                        self._model.train(wikilink, wrong, actual, actual)

        self._model.finilizeTraining()
        print "done training."