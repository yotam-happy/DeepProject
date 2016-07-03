import os
from zipfile import ZipFile
import pandas as pd # pandas
import re
import ujson as json
import cProfile

class WikilinksOldIterator:
    """
    This iterator is meant to be used with the older format of the dataset where each
    file has a single json with many wikilinks in it. It is arguably faster then the
    new iterator but requires quite a lot of memory

    note that WikilinksNewIterator and WikilinksOldIterator can be dropped-in-replacements of each other
    """

    def __init__(self, path="wikilink.zip"):
        """
        :param path:    Path to either a zip file or a directory containing the dataset
        """
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

    def wikilinks(self):
        """
        This is the main function - it is a generator that can be used as an iterator
        returning a single wikilink object at a time
        """
        for f in self._wikilink_files():
            df = pd.read_json(f)
            for wlink in df.wlinks:
                yield wlink
            df = None
            f.close()

class WikilinksNewIterator:
    """
    The new iterator for the version of the dataset where each file contains many jsons,
    each one for a single wikilink and in a single line

    note that WikilinksNewIterator and WikilinksOldIterator can be dropped-in-replacements of each other
    """

    def __init__(self, path, limit_files = 0):
        """
        :param path:        can only be a directory here, no zip file support (caus i'm lazy)
        :param limit_files: if specified then we read only this number of files (good for testing stuff quickly)
        """
        self._path = path
        self._limit_files = limit_files

    def _wikilink_files(self):
        for file in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, file)):
                continue
            print "opening ", file
            yield open(os.path.join(self._path, file), 'r')

    def wikilinks(self):
        """
        This is the main function - it is a generator that can be used as an iterator
        returning a single wikilink object at a time
        """
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
