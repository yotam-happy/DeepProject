import json
import math
import random
import sys
import unicodedata

import nltk
import numpy as np
from nltk.corpus import stopwords
import utils.text

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
        self.conceptCounts2 = dict()
        self.contextDictionary = dict()
        if load_from_file_path is not None:
            self.loadFromFile(load_from_file_path)

        self.conceptLogCountsVariance = np.var([math.log(float(x)) for x in self.conceptCounts.values()])
        self.conceptCountsVariance = np.var([float(x) for x in self.conceptCounts.values()])
        self.conceptCountsSum = sum(self.conceptCounts.values())

        self._stopwords = stopwords.words('english')

    def getCandidateConditionalPrior(self, concept, mention):
        concept = str(concept)
        mention_text = utils.text.strip_wiki_title(mention.mention_text())
        if mention_text not in self.mentionLinks or concept not in self.mentionLinks[mention_text]:
            return 0
        return float(self.mentionLinks[mention_text][concept]) / np.sum(self.mentionLinks[mention_text].values())

    def getCandidatePrior(self, concept, normalized=False, log=False):
        concept = str(concept)
        if not normalized:
            return float(self.conceptCounts[concept]) / self.conceptCountsSum if concept in self.conceptCounts else 0

        # if normalized, normalize by variance
        if log:
            return math.log(float(self.conceptCounts[concept])) / self.conceptLogCountsVariance \
                if concept in self.conceptCounts else 0
        else:
            return float(self.conceptCounts[concept]) / self.conceptCountsVariance \
                if concept in self.conceptCounts else 0

    def getCandidatePriorYamadaStyle(self, entity):
        entity = str(entity)
        return float(self.conceptCounts2[entity]) / len(self.mentionCounts) if entity in self.conceptCounts2 else 0

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
        f.write(json.dumps(self.conceptCounts2)+'\n')
        f.write(json.dumps(self.contextDictionary)+'\n')
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.conceptCounts = json.loads(l[2])
        self.conceptCounts2 = json.loads(l[3])
        self.contextDictionary = json.loads(l[4])
        f.close()

    def calcStatistics(self):
        """
        calculates statistics and populates the class members. This should be called explicitly
        as it might take some time to complete. It is better to call this method once and save
        the results to a file if the dataset is not expected to change
        """
        print "getting statistics"
        for wlink in self._wikilinks_iter.wikilinks():
            mention_text = utils.text.strip_wiki_title(wlink['word'])

            if mention_text not in self.mentionLinks:
                self.mentionLinks[mention_text] = dict()
            self.mentionLinks[mention_text][wlink['wikiId']] = self.mentionLinks[mention_text].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[mention_text] = self.mentionCounts.get(mention_text, 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in wlink['right_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in wlink['left_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

        # counts mentions per concept
        for mention, entities in self.mentionLinks.iteritems():
            for entity in entities.keys():
                self.conceptCounts2[entity] = self.conceptCounts2.get(entity, 0) + 1

    def getCandidatesForMention(self, mention, p=0.001, t=3):
        """
        Returns the most probable sense + all other candidates where p(candidate|mention)>=p
        and with at least t appearances

        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        mention_text = utils.text.strip_wiki_title(
            mention.mention_text() if hasattr(mention, 'mention_text') else mention)
        if mention_text not in self.mentionLinks:
            return {}
        l = self._sortedList(self.mentionLinks[mention_text])
        tot = sum([x[1] for x in l])
        out = dict()
        for x in l:
            if len(out) == 0 or (float(x[1]) / tot >= p and x[1] > t):
                out[int(x[0])] = x[1]

        return {x for x, y in out.iteritems()}

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
        print len(s)
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

#from WikilinksIterator import *
#_path = "/home/yotam/pythonWorkspace/deepProject"
#stats = WikilinksStatistics(WikilinksNewIterator(_path+"/data/intralinks/all"))
#stats.calcStatistics()
#stats.saveToFile(_path + "/data/intralinks/all-stats")
#print "done"