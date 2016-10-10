import json
import math
import random
import sys
import unicodedata

import nltk
import numpy as np
from nltk.corpus import stopwords


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
        self.seenWith = dict()
        self.titleIndex = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()
        if load_from_file_path is not None:
            self.loadFromFile(load_from_file_path)

        self.conceptLogCountsVariance = np.var([math.log(float(x)) for x in self.conceptCounts.values()])
        self.conceptCountsVariance = np.var([float(x) for x in self.conceptCounts.values()])

        self._stopwords = stopwords.words('english')

    def getCandidateProbability(self,concept):
        return float(self.conceptCounts[concept]) / sum(self.conceptCounts.values())

    def getCandidateConditionalProbabilty(self, concept, mention):
        return float(self.mentionLinks[mention][concept]) / np.sum(self.mentionLinks[mention].values())

    def getConceptPrior(self, concept, log=False):
        if log:
            return math.log(float(self.conceptCounts[concept])) / self.conceptLogCountsVariance \
                if concept in self.conceptCounts else 0
        else:
            return float(self.conceptCounts[concept]) / self.conceptCountsVariance \
                if concept in self.conceptCounts else 0

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
        f.write(json.dumps(self.contextDictionary)+'\n')
        f.write(json.dumps(self.titleIndex)+'\n')
#        f.write(json.dumps(self.seenWith))
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.conceptCounts = json.loads(l[2])
        self.contextDictionary = json.loads(l[3])
        self.titleIndex = json.loads(l[4])
#        self.seenWith = json.loads(l[5])
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

        # self.calcCandidatesByPartialTitle()

    def calcMoreStatistics(self):
        self.seenWith = dict()
        # for each sense, count all other senses it was seen with
        for candidates in self.mentionLinks.values():
            for candidate in candidates.keys():
                for other, count in candidates.items():
                    if other != candidate:
                        if candidate not in self.seenWith:
                            self.seenWith[candidate] = dict()
                        self.seenWith[candidate][other] = self.seenWith[candidate].get(other, 0) + count

    def calcCandidatesByPartialTitle(self, db):
        query = "SELECT title, id FROM article"
        self.titleIndex = dict()

        i = 0
        db._cursor.execute(query)
        while True:
            row = db._cursor.fetchone()
            if not row:
                break
            concept_id = int(row[1])
            title = unicodedata.normalize('NFKD', row[0].decode("utf-8")).encode('ascii','ignore')
            title_words = [str(w).lower() for w in nltk.word_tokenize(title)
                           if w not in self._stopwords and len(w) > 2]
            for w in title_words:
                if w not in self.titleIndex:
                    self.titleIndex[w] = dict()
                self.titleIndex[w][concept_id]=1
            i += 1
            if i % 10000 == 0:
                print i

    def ngrams(self, words):
        for k in xrange(len(words)):
            for i in xrange(len(words) - k):
                ngram = " ".join(words[i:i+k+1])
                yield ngram

    def calcCandidatesByPartialTitle3(self, db):
        query = "SELECT title, id FROM article"
        self.titleIndex = dict()

        i = 0
        db._cursor.execute(query)
        while True:
            row = db._cursor.fetchone()
            if not row:
                break
            concept_id = int(row[1])
            title = unicodedata.normalize('NFKD', row[0].decode("utf-8")).encode('ascii','ignore')
            title_words = [str(w).lower() for w in nltk.word_tokenize(title)
                           if w not in self._stopwords and len(w) > 2]
            for k in xrange(len(title_words)):
                ngram = " ".join(title_words[:k+1])
                if ngram not in self.titleIndex:
                    self.titleIndex[ngram] = dict()
                self.titleIndex[ngram][concept_id]=1
            i += 1
            if i % 10000 == 0:
                print i

    def calcCandidatesByRedirectTitle(self, db):
        db.cachePageInfoTable()
        db.cacheArticleTable()

        query = "SELECT title, page_id FROM pages_redirects"
        self.titleIndex = dict()

        i = 0
        db._cursor.execute(query)
        while True:
            row = db._cursor.fetchone()
            if not row:
                break
            concept_id = int(row[1])
            title = unicodedata.normalize('NFKD', row[0].decode("utf-8")).encode('ascii','ignore')
            title_words = [str(w).lower() for w in nltk.word_tokenize(title)
                           if w not in self._stopwords and len(w) > 2]
            for k in xrange(len(title_words)):
                ngram = " ".join(title_words[:k+1])
                if ngram not in self.titleIndex:
                    self.titleIndex[ngram] = dict()
                self.titleIndex[ngram][concept_id]=1
            i += 1
            if i % 10000 == 0:
                print i

    def calcCandidatesByPartialTitle2(self, db):
        query = "SELECT title, id FROM article"
        self.titleIndex = dict()

        i = 0
        db._cursor.execute(query)
        while True:
            row = db._cursor.fetchone()
            if not row:
                break
            concept_id = int(row[1])
            title = unicodedata.normalize('NFKD', row[0].decode("utf-8")).encode('ascii','ignore')
            title_words = [str(w).lower() for w in nltk.word_tokenize(title)
                           if w not in self._stopwords and len(w) > 2]
            for ngram in self.ngrams(title_words):
                if ngram not in self.titleIndex:
                    self.titleIndex[ngram] = dict()
                self.titleIndex[ngram][concept_id]=1
            i += 1
            if i % 10000 == 0:
                print i

    def calcCandidatesByPartialTitle(self, db):
        query = "SELECT title, id FROM article"
        self.titleIndex = dict()

        i = 0
        db._cursor.execute(query)
        while True:
            row = db._cursor.fetchone()
            if not row:
                break
            concept_id = int(row[1])
            title = unicodedata.normalize('NFKD', row[0].decode("utf-8")).encode('ascii','ignore')
            title_words = [str(w).lower() for w in nltk.word_tokenize(title)
                           if w not in self._stopwords and len(w) > 2]
            for w in title_words:
                if w not in self.titleIndex:
                    self.titleIndex[w] = dict()
                self.titleIndex[w][concept_id]=1
            i += 1
            if i % 10000 == 0:
                print i

    def getCandidatesForMention2(self, mention):
        x = self.getCandidatesForMention(mention)

        mention_words = [w for w in nltk.word_tokenize(mention.lower().decode("utf-8"))
                         if w not in self._stopwords and len(w) > 2]
        for word in mention_words:
            if word in self.titleIndex:
                for concept in self.titleIndex[word].keys():
                    if concept not in x:
                        x[int(concept)] = 0.01
        return x

    def getCandidatesForMention3(self, mention):
        x = self.getCandidatesForMention(mention)

        mention_words = [w for w in nltk.word_tokenize(mention.lower().decode("utf-8"))
                         if w not in self._stopwords and len(w) > 2]
        m = " ".join(mention_words)
        if m in self.titleIndex:
            for concept in self.titleIndex[m].keys():
                if concept not in x:
                    x[int(concept)] = 0.01
        return x

    def getCandidatesForMention(self, mention, p=0.001, t=3):
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

    def getCandidatesSeenWith(self, sense, p=0.001, t=3):
        """
        Returns the most probable sense + all other candidates where p(candidate|mention)>=p
        and with at least t appearances

        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        sense = str(sense)
        if sense not in self.seenWith:
            return {}
        l = self._sortedList(self.seenWith[sense])
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

#stats = WikilinksStatistics(None, load_from_file_path="../data/intralinks/train-stats-new2")
#print len(stats.titleIndex)
#from DbWrapper import *
#wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=False)
#stats.calcCandidatesByPartialTitle3(wikiDB)
#stats.saveToFile("../data/intralinks/train-stats-new2")
#print "hi"