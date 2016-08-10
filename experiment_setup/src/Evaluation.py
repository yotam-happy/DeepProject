
class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, model, w2v=None, stats = None, wordExcludeFilter = None, wordIncludeFilter = None):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._iter = test_iter
        self._model = model
        self._stats = stats
        self._wordExcludeFilter = wordExcludeFilter
        self._wordIncludeFilter = wordIncludeFilter
        self._w2v = w2v

        self.n_samples = 0
        self.correct = 0
        self.no_prediction = 0
        self.possible = 0

    def isInStats(self, wlink):
        l = self._stats.getCandidatesForMention(wlink["word"])
        l = {int(x):y for x,y in l.iteritems() if self._w2v is None or int(x) in self._w2v.conceptDict}
        return wlink["wikiId"] in l


    def evaluate(self):
        """
        Do the work - runs over the given test/evaluation set and compares
        the predictions of the model to the actual sense.

        Populates the members:
        self.n_samples:     number of samples we tested on
        self.correct:       number of correct predictions
        self.no_prediction: number of samples the model did not return any prediction for

        :return:
        """
        self.n_samples = 0
        self.correct = 0
        self.no_prediction = 0
        self.possible = 0

        for wikilink in self._iter.wikilinks():
            if self._wordIncludeFilter is not None and wikilink["word"] not in self._wordIncludeFilter:
                continue
            if self._wordExcludeFilter is not None and wikilink["word"] in self._wordExcludeFilter:
                continue
            if 'wikiId' not in wikilink:
                continue
            actual = wikilink['wikiId']
            if self._stats is not None and self.isInStats(wikilink):
                self.possible += 1

            prediction = self._model.predict(wikilink)

            self.n_samples += 1
            if prediction is None:
                self.no_prediction += 1
            elif prediction == actual:
                self.correct += 1

            if(self.n_samples % 1000 == 0):
                print 'sampels=', self.n_samples ,'; %correct=', float(self.correct) / (self.n_samples - self.no_prediction)

        self.printEvaluation()

    def precision(self):
        return float(self.correct) / self.n_samples

    def printEvaluation(self):
        """
        Pretty print results of evaluation
        """
        print "samples: ", self.n_samples, "; correct: ", self.correct, " no-train: ", self.no_prediction, " possible: ", float(self.possible) / self.n_samples
        print "%correct from total: ", float(self.correct) / self.n_samples
        print "%correct where prediction was attempted: ", float(self.correct) / (self.n_samples - self.no_prediction)
