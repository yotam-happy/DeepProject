import pickle

import numpy as np
import os

from WikilinksIterator import WikilinksNewIterator
from WikilinksStatistics import WikilinksStatistics


class Word2vecLoader:
    """
    Words a word2vec model. Loads both word vectors and concept (context) vectors. These must
    match in vector sizes.
    Notice the files are not loaded when the object is created and one must explicitly
    call loadEmbeddings() to do so
    """

    def __init__(self, wordsFilePath="vecs", conceptsFilePath="context"):
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
        with open(path) as f:
            f.readline() # skip embedding size def
            for line in iter(f):
                s = line.split()
                if filterSet is None or s[0] in filterSet:
                    embedding[s[0].lower()] = np.array([float(x) for x in s[1:]])
        return embedding

    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        """
        Loads both word and context embeddings.

        :param wordDict: If specified, only words appearing in word dict will be kept in memory
        :param conceptDict: If specified, only concepts appearing in concept dict will be kept in memory
        """
        parent_path = os.path.abspath(os.path.join(self._wordsFilePath, os.pardir))
        if(os.path.isfile(parent_path+'\\w2v_filt.txt')):
            output = open(parent_path+'\\w2v_filt.txt','rb')
            self.wordEmbeddings = pickle.load(output)
            self.conceptEmbeddings = pickle.load(output)
        else:
            self.wordEmbeddings = self._loadEmbedding(self._wordsFilePath, wordDict)
            self.conceptEmbeddings = self._loadEmbedding(self._conceptsFilePath, conceptDict)

if __name__ == "__main__":
    w2v = Word2vecLoader(wordsFilePath="..\\..\\data\\word2vec\\dim300vecs",
                         conceptsFilePath="..\\..\\data\\word2vec\\dim300context_vecs")
    itr_train = WikilinksNewIterator("..\\..\\data\\wikilinks\\train")
    itr_stats = WikilinksStatistics(itr_train, load_from_file_path="..\\..\\data\\wikilinks\\train_stats")
    wD = itr_stats.mentionLinks
    cD = itr_stats.contextDictionary
    w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
    print 'wordEmbedding size is ',len(w2v.wordEmbeddings)
    print 'conceptEmbeddings size is ',len(w2v.conceptEmbeddings)
    file_w2v = open("..\\..\\data\\word2vec\\w2v_filt.txt",'ab+')
    pickle.dump(w2v.wordEmbeddings, file_w2v)
    pickle.dump(w2v.conceptEmbeddings, file_w2v)
    file_w2v.close()



