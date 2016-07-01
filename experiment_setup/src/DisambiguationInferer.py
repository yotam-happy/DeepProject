class DisambiguationInferer:
    def __init__(self, wikilink_stats=None, w2v=None):
        self._wikilink_stats = wikilink_stats
        self._w2v = w2v

    def baseline_infer(self, wikilink):
        if self._wikilink_stats is None:
            raise Exception("Naive inferer must have statistics object")

        # get statistics for word
        links = self._wikilink_stats.mentionLinks[wikilink["word"]]
        # get most probably sense
        concept = max(links, key=lambda k: k[1])[0]
        return concept
