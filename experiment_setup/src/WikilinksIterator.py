import os
import json
from zipfile import ZipFile
import pandas as pd # pandas
import re

class WikilinksOldIterator:

    # path should either point to a zip file or a directory containing all dataset files,
    def __init__(self, path="wikilink.zip"):
        self._path = path

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

    # the main function - returns a generator that can iterate over all dataset
    def wikilinks(self):
        for f in self._wikilink_files():
            df = pd.read_json(f)
            for wlink in df.wlinks:
                yield wlink
            df = None
            f.close()

class WikilinksNewIterator:

    # path should either point to a zip file or a directory containing all dataset files,
    def __init__(self, path="wikilink.zip"):
        self._path = path

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

    # the main function - returns a generator that can iterate over all dataset
    def wikilinks(self):
        for f in self._wikilink_files():
            for line in f:
                if len(line) > 0:
                    yield json.loads(line)
            f.close()

    # transforms a context into a list of words
    def contextAsList(self, context):
        # Might need more processing?
        return str.split(re.sub(r'\W+', '', context))

class WikilnksStatistics:
    def __init__(self, wikilinks_iter):
        self._wikilinks_iter = wikilinks_iter
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()
        self.calcStatistics()

    # goes over all dataset and calculates a number statistics
    def calcStatistics(self):
        print "getting statistics"
        for wlink in self._wikilinks_iter.wikilinks():
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

if __name__ == "__main__":
    wikilinks = WikilinksIterator("C:\\repo\\WikiLink\\ids")
    stats = WikilnksStatistics(wikilinks)

    def sortedList(l):
        l = [(k,v) for k,v in l.items()]
        l.sort(key=lambda (k,v):-v)
        l.append(("--",0))
        return l

    wordsSorted = [(k, sortedList(v), sum(v.values())) for k,v in stats.mentionLinks.items()]
    wordsSorted.sort(key=lambda (k, v, d): v[1][1])
    print "distinct mentions: ", len(stats.mentionLinks)