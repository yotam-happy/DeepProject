from SimpleW2VPairwiseModel import SimpleW2VPairwiseModel
from WikilinksIterator import *
from BaselinePairwiseModel import *
from KnockoutModel import *
from WikilinksStatistics import *
from Word2vecLoader import *


class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, model):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._iter = test_iter
        self._model = model

        self.n_samples = 0
        self.correct = 0
        self.no_prediction = 0

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
            if 'wikiId' not in wikilink:
                continue
            actual = wikilink['wikiId']
            prediction = self._model.predict(wikilink)

            self.n_samples += 1
            if prediction is None:
                self.no_prediction += 1
            elif prediction == actual:
                self.correct += 1

            if(not self.n_samples.__mod__(1e3)):
                print ' --> ',self.n_samples,' were predicted'

        self.printEvaluation()

    def printEvaluation(self):
        """
        Pretty print results of evaluation
        """
        print "samples: ", self.n_samples, "; correct: ", self.correct, " no-train: ", self.no_prediction
        print "%correct from total: ", float(self.correct) / self.n_samples
        print "%correct where prediction was attempted: ", float(self.correct) / (self.n_samples - self.no_prediction)

if __name__ == "__main__":
    print 'Starts model evluation\nStarts loading files...'
    iter_train = WikilinksNewIterator("..\\..\\data\\wikilinks\\train")
    iter_eval = WikilinksNewIterator("..\\..\\data\\wikilinks\\evaluation")
    train_stats = WikilinksStatistics(iter_train, load_from_file_path="..\\..\\data\\wikilinks\\train_stats")
    w2v = Word2vecLoader(wordsFilePath="..\\..\\data\\word2vec\\dim300vecs",
                         conceptsFilePath="..\\..\\data\\word2vec\\dim300context_vecs")
    wD = train_stats.mentionLinks
    cD = train_stats.conceptCounts

    print 'Load embeddings...'
    w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
    print ' ** wordEmbedding size is ',len(w2v.wordEmbeddings)
    print ' ** conceptEmbeddings size is ',len(w2v.conceptEmbeddings)

    # build model
    # pairwise_model = BaselinePairwiseModel(train_stats)
    # knockout_model = KnockoutModel(pairwise_model, train_stats)

    # build Similarity model
    simple_model = SimpleW2VPairwiseModel(w2v)
    knockout_model = KnockoutModel(simple_model,train_stats)

    print 'Prediction...'
    ev = Evaluation(iter_eval, knockout_model)
    ev.evaluate()
