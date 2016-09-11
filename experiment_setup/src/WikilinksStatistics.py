from WikilinksIterator import WikilinksNewIterator
import json
import random
import sys
import numpy as np
import math

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

        # Is the log justified??? I don't know
        self.conceptCountsMean = np.mean([math.log(x) for x in self.conceptCounts.values()])
        self.conceptCountsVariance = np.var([math.log(x) for x in self.conceptCounts.values()])

    def getConceptPrior(self, concept):
        # Is the log justified??? I don't know
        if concept in self.conceptCounts:
            return (math.log(self.conceptCounts[concept]) - self.conceptCountsMean) / self.conceptCountsVariance
        else:
            return 0

    def getRandomWordSubset(self, p, baseSubset=None):
        '''
        Returns a set with a random subset of the words. p is the size ([0,1])
        :param p:
        :return:
        '''
        if baseSubset is None:
            baseSubset = self.mentionCounts
        return {x for x in baseSubset if random.random() <= p}

    def getSensesFor(self, l):
        return {s for w in l for s in self.getCandidatesForMention(w)}

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
            word = wlink['word'].lower()

            if not word in self.mentionLinks:
                self.mentionLinks[word] = dict()
            self.mentionLinks[word][wlink['wikiId']] = self.mentionLinks[word].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[word] = self.mentionCounts.get(word, 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in wlink['right_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in wlink['left_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

    def getCandidatesForMention(self, mention, p=0.01, t=5):
        """
        Returns the most probable sense + all other candidates where p(candidate|mention)>=p
        and with at least t appearances

        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        if mention.lower() not in self.mentionLinks:
            return {} # TODO: Fix to {}
        l = self._sortedList(self.mentionLinks[mention.lower()])
        tot = sum([x[1] for x in l])
        out = dict()
        for x in l:
            if len(out) == 0 or (float(x[1]) / tot >= p and x[1] > t):
                out[int(x[0])] = x[1]

        # now calc actual priors
        tot = sum([x for x in out.values()])
        out = {x: float(y)/tot for x, y in out.iteritems()}
        return out

    def getGoodMentionsToDisambiguate(self):
        """
        Returns a set of mentions that are deemed "good"
        :param f:
        :return:
        """

        # take those mentions where the second most common term appears more then f times
        s = set()
        for mention in self.mentionLinks:
            l = self.getCandidatesForMention(mention)
            if l is not None and len(l) > 1:
                s.add(mention)
        return s

    def prettyPrintMentionStats(self, m, db):
        try:
            s = "["
            for x,y in m.iteritems():
                t = db.getPageInfoById(x)[2]
                s += str(t) + ": " + str(y) + "; "
            s += ']'
            print s
        except :
            print "Unexpected error:", sys.exc_info()[0]
            print m

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

        k, v = self.mentionLinks.items()[0]
        wordsSorted = [(k, self._sortedList(v), sum(v.values())) for k,v in self.mentionLinks.items()]
        wordsSorted.sort(key=lambda (k, v, d): v[1][1] if len(v) > 1 else 0)

        print("some ambiguous terms:")
        for w in wordsSorted[-10:]:
            print w
