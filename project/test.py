# special IPython command to prepare the notebook for matplotlib

import os
import re
import numpy
import requests
from StringIO import StringIO
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting
import datetime as dt # module for manipulating dates and times
import numpy.linalg as lin # module for performing linear algebra operationsb
from zipfile import ZipFile
import json
import requests

class word2vecLoader:
    def __init__(self, wordsFilePath="vecs", conceptsFilePath="context"):
        # loads words and contexts
        self._wordsFilePath = wordsFilePath
        self._conceptsFilePath = conceptsFilePath

        self.wordEmbeddings = dict()
        self.conceptEmbeddings = dict()

        # make sure embedding sizes match
        with open(wordsFilePath) as f:
            _, self.embeddingSize = f.readline().split()

        with open(conceptsFilePath) as f:
            _, embeddingSz = f.readline().split()
            if embeddingSz != self.embeddingSize:
                raise Exception("Embedding sizes don't match")

    def _loadEmbedding(self, path, filterSet):
        embedding = dict()
        with open(self._wordsFilePath) as f:
            f.readline() # skip embedding size def
            for line in iter(f):
                s = line.split
                if filterSet is None or s[0] in filterSet:
                    embedding[s[0]] = numpy.array([float(x) for x in s[1:]])
        return embedding

    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        self.wordEmbeddings = self._loadEmbedding(self._wordsFilePath, wordDict)
        self.conceptEmbeddings = self._loadEmbedding(self._conceptsFilePath, conceptDict)

class WikilinksGenerator:
    def __init__(self, path="wikilink.zip"):
        self._path = path
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()

    def _wikilink_files(self):
        if os.path.isdir(self._path):
            for file in os.listdir(self._path):
                if os.path.isdir(os.path.join(self._path, file)):
                    continue
                print "opening ", file
                yield open(os.path.join(self._path, file), 'r')
        else: # assume zip
            zf = ZipFile(self._path, 'r') # Read in a list of zipped files
            for fname in zf.namelist():
                print "opening ", fname
                yield zf.open(fname)

    def wikilinks(self):
        for f in self._wikilink_files():
            df = pd.read_json(f)
            for wlink in df.wlinks:
                yield wlink
            df = None
            f.close()

    def contextAsList(self, context):
        # Might need more processing?
        return str.split(re.sub(r'\W+', '', context))

    def calcStatistics(self):
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()
        print "getting statistics"
        for wlink in self.wikilinks():
            if (not 'word' in wlink) or (not 'wikiId' in wlink):
                continue
            if not ('right_context' in wlink or 'left_context' in wlink):
                continue
            if not wlink['word'] in self.mentionLinks:
                self.mentionLinks[wlink['word']] = dict()
            self.mentionLinks[wlink['word']][wlink['wikiId']] = self.mentionLinks[wlink['word']].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[wlink['word']] = self.mentionCounts.get(wlink['word'], 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in self.contextAsList(wlink['right_context']):
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in self.contextAsList(wlink['left_context']):
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

# main commands
if os.path.isdir("C:\\repo\\WikiLink\\ids"):
    path = "C:\\repo\\WikiLink\\ids"
    noam_yotam_flag = 2
elif os.path.isdir("C:\Users\Noam\Documents\Data_DeepESA"):
    path = "C:\Users\Noam\Documents\Data_DeepESA"
    noam_yotam_flag = 1

wikilinks = WikilinksGenerator(path)
words = wikilinks.getWordStatistics()

def sortedList(l):
    l = [(k,v) for k,v in l.items()]
    l.sort(key=lambda (k,v):-v)
    l.append(("--",0))
    return l

wordsSorted = [(k, sortedList(v), sum(v.values())) for k,v in words.items()]
wordsSorted.sort(key=lambda (k, v, d): v[1][1])
print "total words: ", len(words)
