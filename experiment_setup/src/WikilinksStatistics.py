from WikilinksIterator import *

class WikilinksStatistics:
    """
    This class can calculate a number of statistics regarding the
    wikilink dataset.

    To calculate the statistics one needs to call calcStatistics() method.
    The class will then populate the following member variables:

    mentionCounts       dictionary of mention=count. Where mention is a surface term to be disambiguated
                        and count is how many times it was seen n the dataset
    conceptCounts       dictionary of concept=count. Where a concept is a wikipedia id (a sense), and count
                        is how many times it was seen (how many mentions refered to it)
    contextDictionary   dictionary of all words that appeared inside some context (and how many times)
    mentionLinks        holds for each mention a dictionary of conceptsIds it was reffering to and how many
                        times each. (So its a dictionary of dictionaries)
    """

    def __init__(self, wikilinks_iter, load_from_file_path=None):
        """
        Note the statistics are not calculated during init. (unless loaded from file)
        so must explicitly call calcStatistics()
        :param wikilinks_iter:      Iterator to a dataset
        :param load_from_file_path: If given then the statistics are loaded from this file
        """
        self._wikilinks_iter = wikilinks_iter
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()
        if load_from_file_path is not None:
            self.loadFromFile(load_from_file_path)


    def saveToFile(self, path):
        """ saves statistics to a file """
        f = open(path, mode='w')
        f.write(json.dumps(self.mentionCounts)+'\n')
        f.write(json.dumps(self.mentionLinks)+'\n')
        f.write(json.dumps(self.conceptCounts)+'\n')
        f.write(json.dumps(self.contextDictionary))
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.conceptCounts = json.loads(l[2])
        self.contextDictionary = json.loads(l[3])
        f.close()

    def calcStatistics(self):
        """
        calculates statistics and populates the class members. This should be called explicitly
        as it might take some time to complete. It is better to call this method once and save
        the results to a file if the dataset is not expected to change
        """
        print "getting statistics"
        for wlink in self._wikilinks_iter.wikilinks():
            if not wlink['word'] in self.mentionLinks:
                self.mentionLinks[wlink['word']] = dict()
            self.mentionLinks[wlink['word']][wlink['wikiId']] = self.mentionLinks[wlink['word']].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[wlink['word']] = self.mentionCounts.get(wlink['word'], 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in wlink['right_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in wlink['left_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

    def getCandidatesForMention(self, mention):
        """
        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        if mention not in self.mentionLinks:
            return None
        return self.mentionLinks[mention]

    def getGoodMentionsToDisambiguate(self, f=5):
        """
        Returns a set of mentions that are deemed "good"
        These are mentions where the second most common sense appears at least 10 times
        :param f:
        :return:
        """

        # generates a list of mentions, sorted by the second most common sense per
        # mention
        k, v = self.mentionLinks.items()[0]
        l = [(k, self._sortedList(v)) for k,v in self.mentionLinks.items()]

        # take those mentions where the second most common term appears more then f times
        s = set()
        for mention in l:
            if len(mention[1]) > 1 and mention[1][1][1] >= f:
                s.add(mention[0])
        return s

    def _sortedList(self, l):
        l = [(k,v) for k,v in l.items()]
        l.sort(key=lambda (k,v):-v)
        return l

    def printSomeStats(self):
        """
        Pretty printing of some of the statistics in this object
        """

        print "distinct terms: ", len(self.mentionCounts)
        print "distinct concepts: ", len(self.conceptCounts)
        print "distinct context words: ", len(self.contextDictionary)

        k, v = stats.mentionLinks.items()[0]
        wordsSorted = [(k, self._sortedList(v), sum(v.values())) for k,v in stats.mentionLinks.items()]
        wordsSorted.sort(key=lambda (k, v, d): v[1][1] if len(v) > 1 else 0)

        print("some ambiguous terms:")
        for w in wordsSorted[-10:]:
            print w

if __name__ == "__main__":
    iter = WikilinksNewIterator("..\\..\\data\\wikilinks\\train")
    stats = WikilinksStatistics(iter)
    stats.calcStatistics()
    stats.saveToFile('..\\..\\data\\wikilinks\\train_stats')
    stats.printSomeStats()
