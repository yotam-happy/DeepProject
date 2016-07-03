import os
from zipfile import ZipFile
import pandas as pd # pandas
import re
import ujson as json
import pickle
import gzip
import cProfile

class WikilinksOldIterator:

    # path should either point to a zip file or a directory containing all dataset files,
    def __init__(self, path="wikilink.zip", limit_files = 0):
        self._path = path
        self.limit_files =  limit_files
        self.wikilink_fname = 'wikilink.zip'

    def get_wlinks(self):
        # outputs the next wlink piece
        # print "get next()"
        for wlink in self.wikilinks():
            yield wlink

    def _wikilink_files(self):
        if os.path.isdir(self._path):
            for file in os.listdir(self._path):
                # if os.path.isdir(os.path.join(self._path, file)):
                #     continue
                # print "opening ", file
                # yield open(os.path.join(self._path, file), 'r')
                if file != self.wikilink_fname: # assuming zip file
                    continue
                zf = ZipFile(self._path+'\\'+self.wikilink_fname, 'r') # Read in a list of zipped files
                for fname in zf.namelist():
                    print "opening ", fname
                    yield zf.open(fname)

        else: # assume zip
            zf = ZipFile(self._path, 'r') # Read in a list of zipped files
            for fname in zf.namelist():
                print "opening ", fname
                yield zf.open(fname)

    # the main function - returns a generator that can iterate over all dataset
    def wikilinks(self):
        c = 0
        for f in self._wikilink_files():
            df = pd.read_json(f)
            for wlink in df.wlinks:
                if(not 'wikiId' in wlink):
                    continue
                yield wlink
            df = None
            c += 1
            f.close()
            if self.limit_files > 0 and c == self.limit_files:
                print "stoppped at file ", self.limit_files
                break


def save_zip(object, filename, bin = 1):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, bin))
    file.close()


def load_zip(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while 1:
        data = file.read()
        if data == "":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object


class WikilinksStatistics:
    def __init__(self, wikilinks_iter):
        self._wikilinks_iter = wikilinks_iter
        self.mentionCounts = dict()
        self.senseDic = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()

    def senseDicCreation(self):
        # creates the {'word',S = [sensens]} dictionary
        if(os.path.isfile(self._wikilinks_iter._path+'\sense_dict.txt.gz')):
            print("loading sense dictionary form \Data")
            self.senseDic = load_zip(self._wikilinks_iter._path+'\sense_dict.txt.gz')
            # output = open(self._wikilinks_iter._path+'\sense_dict.txt', 'rb')
            # self.senseDic = pickle.load(output)    # 'obj_dict' is a dict object
            # output.close()
        else: # if not, run over all data and create dic
            print "creating sense dictionary \Data"
            for wlink in self._wikilinks_iter.wikilinks():
                if not wlink['word'] in self.senseDic:
                    self.senseDic[wlink['word']] = set([wlink['wikiId']])
                else:
                    self.senseDic[wlink['word']].add(wlink['wikiId'])
            print "done!"
            save_zip(self.senseDic, self._wikilinks_iter._path+'\sense_dict.txt.gz')
            # output = open(self._wikilinks_iter._path+'\sense_dict.txt','ab+')
            # pickle.dump(self.senseDic, output)
            # output.close()

            # need to save the dic for further use

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

if __name__ == "__main__":
    iter = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\train", limit_files=1)
    stats = WikilinksStatistics(iter)
    stats.calcStatistics()
    stats.printSomeStats()

