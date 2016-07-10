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
    def __init__(self, path, limit_files = 0, mention_filter=None):
        self._path = path
        self._limit_files = limit_files
        self._mention_filter = mention_filter

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

                    # filter
                    if (not 'word' in wlink) or (not 'wikiId' in wlink):
                        continue
                    if not ('right_context' in wlink or 'left_context' in wlink):
                        continue
                    if self._mention_filter is not None and wlink['word'] not in self._mention_filter:
                        continue

                    wlink['wikiId'] = int(wlink['wikiId'])

                    # preprocess context
                    if 'right_context' in wlink:
                        r_context = unicodedata.normalize('NFKD', wlink['right_context']).encode('ascii','ignore').lower()
                        wlink['right_context'] = nltk.word_tokenize(r_context)
                    if 'left_context' in wlink:
                        l_context = unicodedata.normalize('NFKD', wlink['left_context']).encode('ascii','ignore').lower()
                        wlink['left_context'] = nltk.word_tokenize(l_context)

                    # return
                    yield wlink

            f.close()
            if self._limit_files > 0 and c >= self._limit_files:
                break

if __name__ == "__main__":
    iter = WikilinksNewIterator("C:\\repo\\DeepProject\\data\\wikilinks\\train", limit_files=1)
    for k in iter.wikilinks():
        break