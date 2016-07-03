from WikilinksStatistics import *

class BaselineModel:
    """
    This baseline model uses the statistics on the dataset to always predict the sense that was
    most common for a mention.
    """

    def __init__(self, iter=None, stats_file=None):
        """
        :param iter:        iterator for the training set
        :param stats_file:  path of a stats file, instead of using the iterator to calculate
                            statistics (takes some time)
        """
        self._wikilink_stats = WikilinksStatistics(iter, load_from_file_path=stats_file)
        if stats_file is None:
            self._wikilink_stats.calcStatistics()

    def predict(self, wikilink):
        """
        takes a wikilink and predicts its correct sense (as we'v said, this is always the most
        common sense for each mention)
        """
        if self._wikilink_stats is None:
            raise Exception("Naive inferer must have statistics object")

        if not (wikilink["word"] in self._wikilink_stats.mentionLinks):
            return None

        # get statistics for word
        links = self._wikilink_stats.mentionLinks[wikilink["word"]]
        # get most probably sense
        concept = max(links, key=lambda k: k[1])
        return concept
