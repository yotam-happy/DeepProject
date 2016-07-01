import numpy as np


class Word2vecLoader:
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
        with open(self._wordsFilePath) as f:
            f.readline() # skip embedding size def
            for line in iter(f):
                s = line.split
                if filterSet is None or s[0] in filterSet:
                    embedding[s[0]] = np.array([float(x) for x in s[1:]])
        return embedding

    # loads both word and concept embeddings into memory.
    # wordDict and conceptDict can be used to filter the loaded embeddings so that only those
    # actually needed are loaded (can be important as these might take a lot of memory
    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        self.wordEmbeddings = self._loadEmbedding(self._wordsFilePath, wordDict)
        self.conceptEmbeddings = self._loadEmbedding(self._conceptsFilePath, conceptDict)
