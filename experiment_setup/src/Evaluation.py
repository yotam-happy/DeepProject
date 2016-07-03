from WikilinksIterator import *
from BaselineInferer import BaselineInferer

class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, inferer):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param inferer:     a model to evaluate
        """
        self._iter = test_iter
        self._inferer = inferer

        self.n_samples = 0
        self.correct = 0
        self.no_train = 0

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

        for wikilink in self._iter.wikilinks():
            actual = wikilink['wikiId']
            infered = self._inferer.infer(wikilink)

            self.n_samples += 1
            if infered is None:
                self.no_prediction += 1
            if infered == actual:
                self.correct += 1

        self.printEvaluation()

    def printEvaluation(self):
        """
        Pretty print results of evaluation
        """
        print "samples: ", self.n_samples, "; correct: ", self.correct, " no-train: ", self.no_train
        print "%correct from total: ", float(self.correct) / self.n_samples
        print "%correct where prediction was attempted: ", float(self.correct) / (self.n_samples - self.no_prediction)

def it(iter):
    c = 0
    for a in iter.wikilinks():
        c += 1
    print c

if __name__ == "__main__":
    iter_train = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\train")
    iter_eval = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\evaluation")

    ev = Evaluation(iter_eval, BaselineInferer(iter_train, stats_file='C:\\repo\\WikiLink\\randomized\\train_stats'))
    ev.evaluate()
