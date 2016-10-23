class CandidatesUsingPPRStats:
    def __init__(self, pprstats, db):
        self._pprstats = pprstats
        self._db = db

    def get_candidates_for_mention(self, mention):
        candidates = set()
        for x in self._pprstats.getCandidateUrlsForMention(mention):
            z = self._db.resolvePage(x[x.rfind('/') + 1:])
            if z is not None:
                candidates.add(z)
        return candidates

    def add_candidates_to_mention(self, mention):
        mention.candidates = self.get_candidates_for_mention(mention)

    def add_candidates_to_document(self, document):
        for mention in document.mentions:
            self.add_candidates_to_mention(mention)

    def getAllCandidateSet(self, it):
        all_cands = set()
        for mention in it.mentions():
            all_cands += self.get_candidates_for_mention(mention)


class CandidatesUsingStatisticsObject:
    def __init__(self, stats):
        self._stats = stats

    def get_candidates_for_mention(self, mention):
        return self._stats.getCandidatesForMention(mention)

    def add_candidates_to_mention(self, mention):
        mention.candidates = self.get_candidates_for_mention(mention)

    def add_candidates_to_document(self, document):
        for mention in document.mentions:
            self.add_candidates_to_mention(mention)

if __name__ == "__main__":
    raise "nnn"