from WikilinksIterator import *
from BaselineInferer import BaselineInferer

class Evaluation:
    def __init__(self, test_iter, inferer):
        self._iter = test_iter
        self._inferer = inferer

        self.n_samples = 0
        self.correct = 0
        self.no_train = 0

    def evaluate(self):
        self.n_samples = 0
        self.correct = 0
        self.no_train = 0

        for wikilink in self._iter.wikilinks():
            actual = wikilink['wikiId']
            infered = self._inferer.infer(wikilink)

            self.n_samples += 1
            if infered is None:
                self.no_train += 1
            if infered == actual:
                self.correct += 1

            if self.n_samples % 100000 == 0:
                print "evaluated ", self.n_samples

        self.printEvaluation()

    def printEvaluation(self):
        print "samples: ", self.n_samples, "; correct: ", self.correct, " no-train: ", self.no_train
        print "%correct from total: ", float(self.correct) / self.n_samples
        print "%correct with train: ", float(self.correct) / (self.n_samples - self.no_train)

if __name__ == "__main__":
    iter_train = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\train")
    iter_eval = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\evaluation")
    ev = Evaluation(iter_eval, BaselineInferer(iter_train))
    ev.evaluate()