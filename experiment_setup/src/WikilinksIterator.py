import os
from zipfile import ZipFile
import pandas as pd # pandas
import re
import ujson as json
import pickle
import gzip
import cProfile
import nltk
import unicodedata

class WikilinksOldIterator:
    """
    This iterator is meant to be used with the older format of the dataset where each
    file has a single json with many wikilinks in it. It is arguably faster then the
    new iterator but requires quite a lot of memory

    note that WikilinksNewIterator and WikilinksOldIterator can be dropped-in-replacements of each other
    """

    # path should either point to a zip file or a directory containing all dataset files,
    def __init__(self, path="wikilink.zip", limit_files = 0):
        """
        :param path:    Path to either a zip file or a directory containing the dataset
        """
        self._path = path
        self.limit_files =  limit_files
        self.wikilink_fname = 'wikilink.zip'

    def get_wlinks(self):
        # outputs the next wlink piece
        # print "get next()"
        for wlink in self._wikilinks_iter.wikilinks():
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

    def wikilinks(self):
        """
        This is the main function - it is a generator that can be used as an iterator
        returning a single wikilink object at a time
        """
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
        for c, f in enumerate(self._wikilink_files()):
            lines = f.readlines()
            for line in lines:
                if len(line) > 0:
                    wlink = json.loads(line)

                    # preprocess
                    if 'right_context' in wlink:
                        wlink['right_context'] = unicodedata.normalize('NFKD', wlink['right_context']).encode('ascii','ignore')
                        #wlink['right_context'].encode('utf-8')
                    if 'left_context' in wlink:
                        wlink['left_context'] = unicodedata.normalize('NFKD', wlink['left_context']).encode('ascii','ignore')
                            #wlink['left_context'].encode('utf-8')

                    # filter
                    if (not 'word' in wlink) or (not 'wikiId' in wlink):
                        continue
                    if not ('right_context' in wlink or 'left_context' in wlink):
                        continue

                    # return
                    yield wlink

            f.close()
            if self._limit_files > 0 and c >= self._limit_files:
                break


    # transforms a context into a list of words
    def contextAsList(self, context):
        return nltk.word_tokenize(context)

if __name__ == "__main__":
    iter = WikilinksNewIterator("C:\\repo\\WikiLink\\randomized\\train", limit_files=1)
