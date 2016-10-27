import numpy as np


class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, model, candidator, stats=None,
                 wordExcludeFilter=None, wordIncludeFilter=None, sampling=None):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._iter = test_iter
        self._model = model
        self._candidator = candidator
        self._wordExcludeFilter = wordExcludeFilter
        self._wordIncludeFilter = wordIncludeFilter
        self._sampling = sampling
        self._stats = stats

        self.n_samples = 0
        self.correct = 0
        self.possible = 0
        self.n_docs = 0
        self.macro_p = 0
        self.candidates = 0
        self.mps_correct = 0
        self.n_docs_for_macro = 0

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
        self.n_docs = 0
        self.n_docs_for_macro = 0
        self.correct = 0
        self.mps_correct = 0
        self.possible = 0
        self.macro_p = 0
        self.candidates = 0

        predictor = self._model.getPredictor()

        for doc in self._iter.documents():
            self._candidator.add_candidates_to_document(doc)
            self.n_docs += 1

            correct_per_doc = 0
            possible_per_doc = 0
            for mention in doc.mentions:
                if self._sampling is not None and np.random.rand() > self._sampling:
                    continue
                if self._wordIncludeFilter is not None and mention.mention_text() not in self._wordIncludeFilter:
                    continue
                if self._wordExcludeFilter is not None and mention.mention_text() in self._wordExcludeFilter:
                    continue

                self.n_samples += 1
                actual = mention.gold_sense_id()
                if actual in mention.candidates:
                    self.candidates += len(mention.candidates)
                    possible_per_doc += 1
                    prediction = predictor.predict(mention)
                    if prediction == actual:
                        correct_per_doc += 1

                    mps = self._stats.getMostProbableSense(mention)
                    if mps == actual:
                        self.mps_correct += 1
                if self.n_samples % 100 == 0:
                    self.printEvaluation()
            self.possible += possible_per_doc
            self.correct += correct_per_doc
            if possible_per_doc > 0:
                self.n_docs_for_macro += 1
                self.macro_p += float(correct_per_doc) / possible_per_doc
        print 'done!'
        self.printEvaluation()

    def mircoP(self):
        return float(self.correct) / self.possible if self.possible > 0 else 'n/a'
    def macroP(self):
        return self.macro_p / self.n_docs_for_macro if self.n_docs_for_macro > 0 else 'n/a'
    def printEvaluation(self):
        """
        Pretty print results of evaluation
        """
        tried = float(self.possible) / self.n_samples if self.n_samples > 0 else 'n/a'
        avg_cands = float(self.candidates) / self.possible if self.possible > 0 else 'n/a'
        micro_p = self.mircoP()
        macro_p = self.macroP()
        mps_correct = float(self.mps_correct) / self.possible if self.possible > 0 else 'n/a'
        print self.n_samples, 'samples in ', self.n_docs, 'docs.', tried, \
            '% mentions tried, avg. candidates per mention:",', avg_cands, \
            '". micro p@1:', micro_p, '% macro p@1:', macro_p, "% mps p@1: ", mps_correct, '%'
