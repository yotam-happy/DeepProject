import utils.text
import json

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
            c = self.get_candidates_for_mention(mention)
            for x in c:
                all_cands.add(x)
        return all_cands


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

    def getAllCandidateSet(self, it):
        all_cands = set()
        for mention in it.mentions():
            c = self.get_candidates_for_mention(mention)
            for x in c:
                all_cands.add(x)
        return all_cands


class CandidatesUsingYago2:
    def __init__(self, wiki_stats):
        self.mentions = dict()
        self.wiki_stats = wiki_stats

    def import_yago(self, directory, db):
        self._cache = dict()

        self.mentions = dict()
        self._import_yago2_file(directory + '/means.tsv', db, False)
        self._import_yago2_file(directory + '/hasFamilyName.tsv', db, True)
        self._import_yago2_file(directory + '/hasGivenName.tsv', db, True)

        # replace sets with arrays
        for mention, candidates in self.mentions.iteritems():
            arr = [x for x in candidates]
            self.mentions[mention] = arr


    def _import_yago2_file(self, path, db, entity_first):
        k = 0
        t = 0
        with open(path) as f:
            print 'importing from path:', path
            for i, line in enumerate(f):
                if i % 10000 == 0 and t > 0:
                    print "done", i, "rows (", float(k) / t, " resolved)"

                tokens = line.split('\t')
                if tokens[2].startswith('wikicategory_') or tokens[2].startswith('wordnet_') or \
                        tokens[2].startswith('geoent_'):
                    continue
                t += 1

                mention = tokens[2] if entity_first else tokens[1]
                entity = tokens[1] if entity_first else tokens[2]
                mention = utils.text.strip_wiki_title(mention.decode('unicode-escape'))
                entity = utils.text.strip_wiki_title(entity.decode('unicode-escape'))

                if entity not in self._cache:
                    entity_id = db.resolvePage(entity)
                    self._cache[entity] = entity_id
                else:
                    entity_id = self._cache[entity]

                if entity_id is not None:
                    if mention not in self.mentions:
                        self.mentions[mention] = dict()
                    k += 1
                    self.mentions[mention][entity_id] = 1

    def save(self, path):
        f = open(path, mode='w')
        f.write(json.dumps(self.mentions)+'\n')

    def load(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentions = json.loads(l[0])

    def get_candidates_for_mention(self, mention):
        mention = utils.text.strip_wiki_title(mention.mention_text())
        s = {int(x) for x in self.mentions[mention]} if mention in self.mentions else set()
        return s.union(self.wiki_stats.getCandidatesForMention(mention, t=10))

    def add_candidates_to_mention(self, mention):
        mention.candidates = self.get_candidates_for_mention(mention)

    def add_candidates_to_document(self, document):
        for mention in document.mentions:
            self.add_candidates_to_mention(mention)

    def getAllCandidateSet(self, it):
        all_cands = set()
        k = 0
        jj = 0
        kkk = 0
        for mention in it.mentions():
            c = self.get_candidates_for_mention(mention)
            k += len(c)
            if len(c) > 10000:
                print mention.mention_text(), len(c)
                kkk += 1
            jj += 1
            for x in c:
                all_cands.add(x)
        print float(k) / jj
        print kkk
        return all_cands


if __name__ == "__main__":
    from DbWrapper import *
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    # wikiDB = WikipediaDbWrapper(user='noambox', password='ncTech#1', database='wiki20151002')
    yago2 = CandidatesUsingYago2()
    yago2.import_yago("../data/yago2", wikiDB)
    yago2.save("../data/yago2/yago.candidates")
