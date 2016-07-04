from WikilinksStatistics import *

class SimpleW2VPairwiseModel:
    """
    This is a simple model that decides which of two candidates is better by simply comparing
    the distance between the w2v vectors of each of the candidates to the w2v vector of the context
    The w2v vector for the context is calculated simply as the average of all context words
    """

    def __init__(self, stats, w2v):
        self._stats = stats
        self._w2v = w2v

    def predict(self, wikilink, candidate1, candidate2):
        """
        The prediction function of a pairwise model
        :return:    returns one of the two candidates - the one deemed better
                    This function can returns None if it has nothing to say on either of the candidates
                    this is considered as saying they are both very unlikely and should be eliminated
        """
        return None
