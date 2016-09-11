class BaselinePairwiseModel:
    """
    This baseline model uses the statistics on the dataset to always predict the sense that was
    most common for a mention.
    """

    def __init__(self, stats):
        """
        :param stats: statistics object
        """
        self._stats = stats
        self.cc = 0

    def predict(self, wikilink, candidate1, candidate2):
        """
        The prediction function of a pairwise model
        :return:    returns one of the two candidates - the one deemed better
                    This function can returns None if it has nothing to say on either of the candidates
                    this is considered as saying they are both very unlikely and should be eliminated
        """
        l = self._stats.getCandidatesForMention(wikilink["word"])
        l = {int(x):y for x,y in l.iteritems()}
        if l is None:
            return None

        if candidate1 not in l and candidate2 not in l:
            return None
        elif candidate1 not in l:
            return candidate2
        elif candidate2 not in l:
            return candidate1
        return candidate1 if l[candidate1] > l[candidate2] else candidate2
