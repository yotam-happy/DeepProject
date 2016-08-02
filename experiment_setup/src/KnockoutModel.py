from WikilinksStatistics import *
import random
import nltk
from nltk.corpus import stopwords


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

    def predict2(self, l_context, word, r_context, db):
        _stopwords = stopwords.words('english')

        l_context = [w for w in nltk.word_tokenize(l_context) if w not in _stopwords]
        r_context = [w for w in nltk.word_tokenize(r_context) if w not in _stopwords]
        wikilink = {"word": word, "left_context": l_context, "right_context": r_context}
        id = self.predict(wikilink)
        print str(id), " : ", db.getPageInfoById(id)

    def predict(self, wikilink):
        candidates = self._stats.getCandidatesForMention(wikilink["word"])
        if candidates is None:
            return None

        # do a knockout
        l = [int(candidate) for candidate in candidates.keys()]
        random.shuffle(l)

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
