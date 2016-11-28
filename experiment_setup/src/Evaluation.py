import numpy as np
import time

class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, model, candidator, stats=None,
                 wordExcludeFilter=None, wordIncludeFilter=None,
                 sampling=None, log_path=None, db=None, trained_mentions=None):
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
        self._log_path = log_path
        self._db = db
        self._trained_mentions = trained_mentions

        self.error_from_untrained_mention = 0

        self.acc_by_mention = dict()
        self.acc_by_sense = dict()
        self.confusion_matrix = dict()

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
        self.unique = 0

        self.acc_by_mention = dict()
        self.acc_by_sense = dict()

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
                    if len(mention.candidates) == 1:
                        self.unique += 1
                    self.candidates += len(mention.candidates)
                    possible_per_doc += 1
                    with open('feature_set.txt', 'a') as f:
                        f.write('\n' + mention.mention_text() + ' (' + str(actual) + ')\n')
                    prediction = predictor.predict(mention)
                    self.collect_advanced_stats(mention, actual, prediction)
                    if prediction == actual:
                        correct_per_doc += 1
                    else:
                        if self._trained_mentions is not None and mention.mention_text() not in self._trained_mentions:
                            self.error_from_untrained_mention += 1
                    mps = self._stats.getMostProbableSense(mention)
                    if mps == actual:
                        self.mps_correct += 1
            if self.n_docs % 10 == 0:
                self.printEvaluation()
            self.possible += possible_per_doc
            self.correct += correct_per_doc
            if possible_per_doc > 0:
                self.n_docs_for_macro += 1
                self.macro_p += float(correct_per_doc) / possible_per_doc
        print 'done!'
        self.printEvaluation()
        self.print_advanced_stats()
        if self._log_path is not None:
            self.saveEvaluation()

    def collect_advanced_stats(self, mention, actual, predicted):
        m = mention.mention_text()
        m_correct, m_total = self.acc_by_mention[m] if m in self.acc_by_mention else (0, 0)
        m_correct += 1 if actual == predicted else 0
        m_total += 1
        self.acc_by_mention[m] = (m_correct, m_total)

        s_correct, s_total = self.acc_by_sense[actual] if actual in self.acc_by_sense else (0, 0)
        s_correct += 1 if actual == predicted else 0
        s_total += 1
        self.acc_by_sense[actual] = (s_correct, s_total)

        if actual not in self.confusion_matrix:
            self.confusion_matrix[actual] = dict()
        if predicted not in self.confusion_matrix[actual]:
            self.confusion_matrix[actual][predicted] = 0
        self.confusion_matrix[actual][predicted] += 1

    def print_advanced_stats(self, confusion=True):

        micro_maybe = sum([float(y[0])/y[1] for x, y in self.acc_by_mention.iteritems()]) /len(self.acc_by_mention)
        print "microrrrr??", micro_maybe
        if self._trained_mentions is not None:
            print "error from untrained mentions", float(self.error_from_untrained_mention) / (self.possible - self.correct)
        print "unique", self.unique / float(self.possible)
        print "possible", self.possible

        acc_by_mention = [(x, float(y[0])/y[1]) for x, y in self.acc_by_mention.iteritems()]
        acc_by_mention.sort(key=lambda (k, v): v)
        s = 'accuracy by mention: '
        for x, y in acc_by_mention:
            s+= x + ': ' + str(y) + '; '
        s += '\n'

        s += 'accuracy by sense: '
        acc_by_sense = [(x, float(y[0]) / y[1]) for x, y in self.acc_by_sense.iteritems()]
        acc_by_sense.sort(key=lambda (k, v): v)
        for x, y in acc_by_sense:
            s += str(self._db.getPageTitle(x)) + '(' + str(x) + '): ' + str(y) + '; '
        s += '\n'

        if confusion:
            s += 'confusion matrix:\n'
            confusion_m = [(x, y) for x, y in self.confusion_matrix.iteritems()]
            confusion_m.sort(key=lambda (k, v): float(self.acc_by_sense[k][0]) / self.acc_by_sense[k][1])
            for x, y in confusion_m:
                a = [(z, w) for z, w in y.iteritems()]
                a.sort(key=lambda (k, v): v)
                ss = str(self._db.getPageTitle(x)) + '(' + str(x) + '): '
                for z, w in a:
                    ss += '[' + str(self._db.getPageTitle(z)) + '(' + str(z) + '),' + str(w) + '], '
                ss += '\n'
                s += ss
        return s


    def mircoP(self):
        return float(self.correct) / self.possible if self.possible > 0 else 'n/a'
    def macroP(self):
        return self.macro_p / self.n_docs_for_macro if self.n_docs_for_macro > 0 else 'n/a'
    def printEvaluation(self):
        print self.evaluation()

    def saveEvaluation(self):
        with open(self._log_path, "a") as f:
            f.write(self.evaluation(advanced=True))
        print self.evaluation()

    def evaluation(self, advanced=False):
        """
        Pretty print results of evaluation
        """
        tried = float(self.possible) / self.n_samples if self.n_samples > 0 else 'n/a'
        avg_cands = float(self.candidates) / self.possible if self.possible > 0 else 'n/a'
        micro_p = self.mircoP()
        macro_p = self.macroP()
        mps_correct = float(self.mps_correct) / self.possible if self.possible > 0 else 'n/a'
        s = time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + ": " + \
            str(self.n_samples) + ' samples in ' + str(self.n_docs) + ' docs. ' + str(tried) + \
            '% mentions tried, avg. candidates per mention:", ' + str(avg_cands) + \
            ' ". micro p@1: ' + str(micro_p) + '% macro p@1: ' + str(macro_p) + "% mps p@1: " + str(mps_correct) + '%\n'
        if advanced:
            s += self.print_advanced_stats()
        return s