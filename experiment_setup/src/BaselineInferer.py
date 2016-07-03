from WikilinksIterator import *

class BaselineInferer:
    def __init__(self, iter=None, stats_file=None):
        self._wikilink_stats = WikilinksStatistics(iter, load_from_file_path=stats_file)
        if stats_file is None:
            self._wikilink_stats.calcStatistics()

    def infer(self, wikilink):
        if self._wikilink_stats is None:
            raise Exception("Naive inferer must have statistics object")

        if not (wikilink["word"] in self._wikilink_stats.mentionLinks):
            return None

        # get statistics for word
        links = self._wikilink_stats.mentionLinks[wikilink["word"]]
        # get most probably sense
        concept = max(links, key=lambda k: k[1])
        return concept
