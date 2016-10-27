import operator

class PointwisePredict:
    """
    This model takes a pairwise model that can train/predict on pairs of candidates for a wikilink
    and uses it to train/predict on a list candidates using a knockout method.
    The candidates are taken from a stats object
    """

    def __init__(self, pointwise_model):
        """
        :param pairwise_model:  The pairwise model used to do prediction/training on a triplet
                                (wikilink,candidate1,candidate2)
        :param stats:           A statistics object used to get list of candidates
        """
        self._pointwise_model = pointwise_model

    def predict(self, mention):

        if len(mention.candidates) < 1:
            return None
        d = {candidate: self._predict(mention, candidate) for candidate in mention.candidates.keys()}
        return max(d.iteritems(), key=operator.itemgetter(1))[0]

    def _predict(self, mention, candidate):
        a = self._pointwise_model.predict(mention, candidate, None)
        return a if a is not None else 0
