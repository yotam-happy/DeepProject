import json
import os
import random

from WikilinksIterator import *

class WikilinksRewrite:
    """
        This class takes an iterator of the dataset and creates a new dataset where
        each wikilink is its own json contained in one line.
        This is opposed to the old style we had where the entire file was a single
        json and each wikilink was in multiple lines.

        wikilinks_iter -    an iterator of the dataset, supposed to be of the old
                            style so this class converts to new style
    """

    def __init__(self, wikilinks_iter, dest_dir, json_per_file=400000):
        self._iter = wikilinks_iter
        self._dest_dir = dest_dir
        self._n = 0
        self._json_per_file = json_per_file

    def _open_file(self, n):
        return open(os.path.join(self._dest_dir, 'wikilinks_{}.json'.format(n)), mode='w')

    def work(self):
        l = []
        for wikilink in self._iter.wikilinks():
            l.append(json.dumps(wikilink))
            if len(l) >= self._json_per_file:
                # write list to file
                f = self._open_file(self._n)
                for s in l:
                    f.write(s + '\n')
                f.close()
                self._n += 1
                l = []


class ShuffleFiles:
    """
    This class takes a source directory which is assumed to contain some text file.
    It then writes the contents of these files into dest_dir, into a similar number
    of files but with the lines: roughly equally devided between the files and randomly
    shuffled both between the files and inside the files.

    the process is a two step process and one must call work1() and then work2() to do
    the job
    """
    def __init__(self, src_dir, dest_dir):
        self._src_dir = src_dir
        self._dest_dir = dest_dir

    def _open_for_write(self, dir, n):
        return open(os.path.join(dir, 'wikilinks_{}.json'.format(n)), mode='w')

    # step 1 of randomizing
    def work1(self):
        # open files for write
        dest_files = [self._open_for_write(self._dest_dir, n) for n in xrange(len(os.listdir(self._src_dir)))]
        print "first phase..."

        for fname in os.listdir(self._src_dir):
            in_f = open(os.path.join(self._src_dir, fname), 'r')
            dest_files_temp = [[] for n in xrange(len(dest_files))]
            for line in in_f:
                dest_files_temp[random.randrange(len(dest_files))].append(line)
            in_f.close()

            for f, l in zip(dest_files, dest_files_temp):
                    f.writelines(l)
            print "done ", fname

        for f in dest_files:
            f.close()

    # step 2 of randomizing
    def work2(self):
        print "second phase..."
        for fname in os.listdir(self._dest_dir):
            print "opening file: " + fname
            f = open(os.path.join(self._dest_dir, fname), 'r')
            l = f.readlines()
            f.close()

            random.shuffle(l)

            f = open(os.path.join(self._dest_dir, fname), 'w')
            f.writelines(l)
            f.close()


if __name__ == "__main__":
    # converts the old format (one json with many wikilinks per file)
    # to new format (one json one single wikilink per line)
#    old_iter = WikilinksOldIterator(path="C:\\repo\\WikiLink\\with_ids")
#    rewriter = WikilinksRewrite(old_iter, "C:\\repo\\WikiLink\\new_format")
#    rewriter.work()

    # randomizes lines in dataset files
    random.seed()
    shuffler = ShuffleFiles('C:\\repo\\WikiLink\\new_format', 'C:\\repo\\WikiLink\\randomized')
    shuffler.work1()
    shuffler.work2()
