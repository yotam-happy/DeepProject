import pickle
from sklearn.preprocessing import normalize
from scipy import spatial
import numpy as np
import os

from WikilinksIterator import WikilinksNewIterator
from WikilinksStatistics import WikilinksStatistics
import keras as K
from yamada.opennlp import *

DUMMY_KEY = '~@@dummy@@~'


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
        self.parser = OpenNLP()

        self.wordEmbeddings = None
        self.wordDict = dict()
        self.wordEmbeddingsSz = 0
        self.conceptEmbeddings = None
        self.conceptDict = dict()
        self.conceptEmbeddingsSz = 0

    def similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_nouns(self, sentence_list):
        nouns = []
        for sent in sentence_list:
            nouns += self.parser.list_nouns(sent)
        return nouns

    def get_entity_vec(self, entity):
        return self.conceptEmbeddings[self.conceptDict[entity], :] if entity in self.conceptDict else None

    def text_to_embedding(self, words, mention):
        vecs = []
        for word in words:
            if not mention.lower().find(word.lower()) > -1 and \
                    word.lower() in self.wordDict:
                vecs.append(self.wordEmbeddings[self.wordDict[word.lower()], :])
        vecs = np.array(vecs)
        return vecs.mean(0)

    def _loadEmbedding(self, path, filterSet, int_key=False):
        with open(path) as f:
            dict_sz, embd_sz = f.readline().split()
            dict_sz = int(dict_sz) if filterSet is None or int(dict_sz) < len(filterSet) else len(filterSet)
            embd_sz = int(embd_sz)

            embd_dict = dict()
            embedding = np.zeros((dict_sz + 2, embd_sz))

            # Adds a dummy key. The dummy is a (0,...,0) vector
            embd_dict[DUMMY_KEY] = 1

            i = 2
            for line in iter(f):
                s = line.split()
                if s[0] == '</s>':
                    continue
                key = int(s[0]) if int_key else s[0].lower()
                if filterSet is None or key in filterSet:
                    embedding[i, :] = np.array([float(x) for x in s[1:]])
                    embd_dict[key] = i
                    i += 1
            embedding = normalize(embedding)
        return embedding, embd_dict, embd_sz

    def _randomEmbedding(self, path, filterSet, int_key = False):
        with open(path) as f:
            dict_sz, embd_sz = f.readline().split()
            dict_sz = int(dict_sz) if filterSet is None or int(dict_sz) < len(filterSet) else len(filterSet)
            embd_sz = int(embd_sz)

            embd_dict = dict()
            embd_dict[DUMMY_KEY] = 1

            embedding = np.random.uniform(-1 / np.sqrt(embd_sz * 4), 1 / np.sqrt(embd_sz * 4), (dict_sz+1,embd_sz))
            embedding[0, :] = np.zeros((1, embd_sz))
            embedding[1, :] = np.zeros((1, embd_sz))

            print "rnd embd"
            i = 2
            for line in iter(f):
                s = line.split()
                if s[0] == '</s>':
                    continue
                key = int(s[0]) if int_key else s[0].lower()
                if filterSet is None or key in filterSet:
                    embd_dict[key] = i
                    i += 1
            return embedding, embd_dict, embd_sz

    def randomEmbeddings(self, wordDict=None, conceptDict=None):
        self.wordEmbeddings, self.wordDict, self.wordEmbeddingsSz = \
            self._randomEmbedding(self._wordsFilePath, wordDict)
        self.conceptEmbeddings, self.conceptDict, self.conceptEmbeddingsSz = \
            self._randomEmbedding(self._conceptsFilePath, conceptDict, int_key=True)


    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        """
        Loads both word and context embeddings.

        :param wordDict: If specified, only words appearing in word dict will be kept in memory
        :param conceptDict: If specified, only concepts appearing in concept dict will be kept in memory
        """
        print "loading word embeddings...", self._conceptsFilePath
        self.wordEmbeddings, self.wordDict, self.wordEmbeddingsSz = \
            self._loadEmbedding(self._wordsFilePath, wordDict)

        print "loading concept embeddings..."
        self.conceptEmbeddings, self.conceptDict, self.conceptEmbeddingsSz = \
            self._loadEmbedding(self._conceptsFilePath, conceptDict, int_key=True)

if __name__ == "__main__":
    w2v = Word2vecLoader(wordsFilePath="..\\..\\data\\word2vec\\dim300vecs",
                         conceptsFilePath="..\\..\\data\\word2vec\\dim300context_vecs")
    itr_train = WikilinksNewIterator("..\\..\\data\\wikilinks\\train")
    itr_stats = WikilinksStatistics(itr_train, load_from_file_path="..\\..\\data\\wikilinks\\train_stats2")
    wD = itr_stats.contextDictionary
    cD = itr_stats.conceptCounts
    w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
    print 'wordEmbedding size is ',len(w2v.wordEmbeddings)
    print 'conceptEmbeddings size is ',len(w2v.conceptEmbeddings)
