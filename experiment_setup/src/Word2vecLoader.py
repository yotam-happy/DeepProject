import numpy as np
import os

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
                    embedding[s[0]] = np.array([float(x) for x in s[1:]])
        return embedding

    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        """
        Loads both word and context embeddings.

        :param wordDict: If specified, only words appearing in word dict will be kept in memory
        :param conceptDict: If specified, only concepts appearing in concept dict will be kept in memory
        """

        self.wordEmbeddings = self._loadEmbedding(self._wordsFilePath, wordDict)
        self.conceptEmbeddings = self._loadEmbedding(self._conceptsFilePath, conceptDict)

if __name__ == "__main__":
    w2v = Word2vecLoader(wordsFilePath="..\\..\\data\\word2vec\\dim300vecs",
                         conceptsFilePath="..\\..\\data\\word2vec\\dim300context_vecs")
    w2v.loadEmbeddings()
    print w2v.wordEmbeddings