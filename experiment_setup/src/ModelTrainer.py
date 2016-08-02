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

    def __init__(self, iter, stats, model, epochs = 10, wordFilter = None,  senseFilter = None):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._neg_sample = 10
        self._iter = iter
        self._model = model
        self._stats = stats
        self._epochs = epochs
        self.wordFilter = wordFilter
        self.senseFilter = senseFilter
        if senseFilter is not None:
            self.senseFilterInt = {int(x) for x in senseFilter}

        self.n_samples = 0

    def train(self):
        print "start training..."
        self._model.startTraining()

        for epoch in xrange(self._epochs):
            print "training epoch ", epoch

            for wikilink in self._iter.wikilinks():
                if self.wordFilter is not None and wikilink["word"] in self.wordFilter:
                    continue

                actual = wikilink['wikiId']
                if self.senseFilter is not None and actual in self.senseFilterInt :
                    continue

                candidates = self._stats.getCandidatesForMention(wikilink["word"])
                if (self.senseFilter is not None):
                    candidates = {x:y for x,y in candidates.iteritems() if x not in self.senseFilter}

                if candidates is None or len(candidates) < 2:
                    continue

                # get probability vector. Added some smoothing
                # TODO: uniform might be better
                ids = [ int(candidate[0]) for candidate in candidates.items() if int(candidate[0]) != actual]
                probs = [ candidate[1] for candidate in candidates.items() if int(candidate[0]) != actual]
                t = float(sum(probs))
                smooth = t / len(probs)
                probs = [(float(p) + smooth) / (t*2) for p in probs]

                for k in xrange(self._neg_sample):
                    wrong = ids[np.random.choice(np.arange(len(probs)), p=probs)]
                    # train on both sides so we get a symmetric model
                    if random.randrange(2) == 0:
                        self._model.train(wikilink, actual, wrong, actual)
                    else:
                        self._model.train(wikilink, wrong, actual, actual)

                # print 'breaked' # DEBUG
                # break

        self._model.finilizeTraining()
        print "done training."