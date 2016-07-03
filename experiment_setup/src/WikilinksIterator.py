import os
from zipfile import ZipFile
import pandas as pd # pandas
import re
import ujson as json
import cProfile

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

    # the new iterator does not support using a zip file.
    def __init__(self, path, limit_files = 0):
        self._path = path
        self._limit_files = limit_files

    def _wikilink_files(self):
        for file in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, file)):
                continue
            print "opening ", file
            yield open(os.path.join(self._path, file), 'r')

    def wikilinks(self):
        c = 0
        for f in self._wikilink_files():
            lines = f.readlines()
            for line in lines:
                if len(line) > 0:
                    wlink = json.loads(line)

                    # preprocess
                    if 'right_context' in wlink:
                        wlink['right_context'] = wlink['right_context'].encode('utf-8')
                    if 'left_context' in wlink:
                        wlink['left_context'] = wlink['left_context'].encode('utf-8')

                    # filter
                    if (not 'word' in wlink) or (not 'wikiId' in wlink):
                        continue
                    if not ('right_context' in wlink or 'left_context' in wlink):
                        continue

                    # return
                    yield wlink

            f.close()
            c += 1
            if self._limit_files > 0 and c == self._limit_files:
                break

    # transforms a context into a list of words
    def contextAsList(self, context):
        # Might need more processing?
        return str.split(re.sub(r'\W+', '', context))

class WikilinksStatistics:
    def __init__(self, wikilinks_iter, load_from_file_path=None):
        self._wikilinks_iter = wikilinks_iter
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()
        if load_from_file_path is not None:
            self.loadFromFile(load_from_file_path)


    def saveToFile(self, path):
        f = open(path, mode='w')
        f.write(json.dumps(self.mentionCounts)+'\n')
        f.write(json.dumps(self.mentionLinks)+'\n')
        f.write(json.dumps(self.conceptCounts)+'\n')
        f.write(json.dumps(self.contextDictionary))
        f.close()

    def loadFromFile(self, path):
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.conceptCounts = json.loads(l[2])
        self.contextDictionary = json.loads(l[3])
        f.close()

    # goes over all dataset and calculates a number statistics
    def calcStatistics(self):
        print "getting statistics"
        for wlink in self._wikilinks_iter.wikilinks():
            if not wlink['word'] in self.mentionLinks:
                self.mentionLinks[wlink['word']] = dict()
            self.mentionLinks[wlink['word']][wlink['wikiId']] = self.mentionLinks[wlink['word']].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[wlink['word']] = self.mentionCounts.get(wlink['word'], 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in self._wikilinks_iter.contextAsList(wlink['right_context']):
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in self._wikilinks_iter.contextAsList(wlink['left_context']):
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

    def _sortedList(self, l):
        l = [(k,v) for k,v in l.items()]
        l.sort(key=lambda (k,v):-v)
        l.append(("--",0))
        return l

    def printSomeStats(self):
        print "distinct terms: ", len(self.mentionCounts)
        print "distinct concepts: ", len(self.conceptCounts)
        print "distinct context words: ", len(self.contextDictionary)

        k, v = stats.mentionLinks.items()[0]
        wordsSorted = [(k, self._sortedList(v), sum(v.values())) for k,v in stats.mentionLinks.items()]
        wordsSorted.sort(key=lambda (k, v, d): v[1][1])

        print("some ambiguous terms:")
        for w in wordsSorted[-10:]:
            print w

if __name__ == "__main__":
    iter = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\train")
    stats = WikilinksStatistics(iter, load_from_file_path='C:\\repo\\WikiLink\\randomized\\train_stats')
    stats.calcStatistics()
    stats.saveToFile('C:\\repo\\WikiLink\\randomized\\train_stats')
    stats.printSomeStats()

