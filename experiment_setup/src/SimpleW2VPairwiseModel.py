from nltk import word_tokenize
from Word2vecLoader import *
from scipy import spatial
from WikilinksStatistics import *

class SimpleW2VPairwiseModel:
    """
    This is a simple model that decides which of two candidates is better by simply comparing
    the distance between the w2v vectors of each of the candidates to the w2v vector of the context
    The w2v vector for the context is calculated simply as the average of all context words
    """

    def __init__(self, stats, w2v):
        self._w2v = w2v

    def predict(self, wikilink, candidate1, candidate2):
        """
        The prediction function of a pairwise model
        :return:    returns one of the two candidates - the one deemed better
                    This function can returns None if it has nothing to say on either of the candidates
                    this is considered as saying they are both very unlikely and should be eliminated
        """
        context = word_tokenize(wikilink['right_context'] + wikilink['left_context'])
        context_vec = [self._w2v.wordEmbeddings[word.lower()] for word in context if word.isalpha() and self._w2v.wordEmbeddings.has_key(word.lower())]
        if((not self._w2v.conceptEmbeddings.has_key(candidate1.lower())) or
               (not self._w2v.conceptEmbeddings.has_key(candidate1.lower())) or
               (not context_vec)):
            return None
        else:
            candidate1_vec = self._w2v.conceptEmbeddings[candidate1.lower()]
            candidate2_vec = self._w2v.conceptEmbeddings[candidate2.lower()]
            mean_context = np.mean(np.asarray(context_vec), 0)
            similarity = findCosineDist(mean_context, candidate1_vec, candidate2_vec)
            return candidate1 if similarity['candidate1'] > similarity['candidate2'] else candidate2



def findCosineDist(w ,c1, c2):
    out = dict()
    out['candidate1'] = spatial.distance.cosine(w,c1)
    out['candidate2'] = spatial.distance.cosine(w,c2)
    return out
