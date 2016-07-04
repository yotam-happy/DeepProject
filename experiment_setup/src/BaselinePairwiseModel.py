class BaselinePairwiseModel:
    """
    This baseline model uses the statistics on the dataset to always predict the sense that was
    most common for a mention.
    """

    def __init__(self, stats):
        """
        :param stats: statistics object
        """
        self._wikilink_stats = stats

    def predict(self, wikilink, candidate1, candidate2):
        """
        The prediction function of a pairwise model
        :return:    returns one of the two candidates - the one deemed better
                    This function can returns None if it has nothing to say on either of the candidates
                    this is considered as saying they are both very unlikely and should be eliminated
        """
        if wikilink["word"] not in self._wikilink_stats.mentionLinks:
            return None

        l = self._wikilink_stats.mentionLinks[wikilink["word"]]

        if candidate1 not in l and candidate2 not in l:
            return None
        elif candidate1 not in l:
            return candidate2
        elif candidate2 not in l:
            return candidate1

        return candidate1 if l[candidate1] > l[candidate2] else candidate2
