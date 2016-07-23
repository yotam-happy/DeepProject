import pickle

from scipy import spatial
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

        self.wordEmbeddings = None
        self.wordDict = dict()
        self.wordEmbeddingsSz = 0
        self.conceptEmbeddings = None
        self.conceptDict = dict()
        self.conceptEmbeddingsSz = 0


    def _loadEmbedding(self, path, filterSet, int_key = False):
        with open(path) as f:
            dict_sz, embd_sz = f.readline().split()
            dict_sz = int(dict_sz) if int(dict_sz) < len(filterSet) else len(filterSet)
            embd_sz = int(embd_sz)

            embd_dict = dict()
            embedding = np.zeros((dict_sz+1,embd_sz))

            print "dict: ", dict_sz, " e", embedding.shape
            i = 1
            for line in iter(f):
                s = line.split()
                if filterSet is None or s[0] in filterSet:
                    embedding[i,:] = np.array([float(x) for x in s[1:]])
                    embd_dict[int(s[0].lower()) if int_key else s[0].lower()] = i
                    i += 1
        return embedding, embd_dict, embd_sz

    def _loadEmbeddingDump(self, np_array_path, dict_path):
        '''
        Loads processed embeddings that were saved
        '''
        dict_file = open(dict_path,'rb')
        embd_dict = pickle.load(dict_file)
        embd_sz = pickle.load(dict_file)
        dict_file.close()
        embd = np.load(np_array_path)
        return (embd, embd_dict, embd_sz)

    def _saveEmbeddingDump(self, np_array_path, dict_path, embd, embd_dict, embd_sz):
        try:
            dict_file = open(dict_path,'wb')
            pickle.dump(self.wordEmbeddings, dict_file)
            pickle.dump(embd_dict, dict_file)
            pickle.dump(embd_sz, dict_file)
            dict_file.close()
            np.save(np_array_path, embd)
        except:
            print "couldn't load embeddings... continue"


    def wordListToVectors(self, l):
        l = []
        for w in l:
            if w in self.wordDict:
                l.append(self.wordEmbeddings[self.wordDict[w]])
        ar = np.asarray(l)
        print len(l)
        print ar.shape
        return ar

    def meanOfWordList(self, l):
        sum = np.zeros(self.embeddingSize)
        k = 0
        for w in l:
            if w in self.wordDict:
                sum += self.wordEmbeddings[self.wordDict[w],:]
                k += 1
        return sum / k

    def distance(self, v1, v2):
        return spatial.distance.cosine(v1,v2)

    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        """
        Loads both word and context embeddings.

        :param wordDict: If specified, only words appearing in word dict will be kept in memory
        :param conceptDict: If specified, only concepts appearing in concept dict will be kept in memory
        """
        print "loading word embeddings...", self._conceptsFilePath
        if os.path.isfile(self._wordsFilePath+'.preprocessed.dict') \
                and os.path.isfile(self._wordsFilePath+'.preprocessed.npy'):
            self.wordEmbeddings, self.wordDict, self.wordEmbeddingsSz = \
                self._loadEmbeddingDump(self._wordsFilePath+'.preprocessed.npy', self._wordsFilePath+'.preprocessed.dict')
        else:
            self.wordEmbeddings, self.wordDict, self.wordEmbeddingsSz = \
                self._loadEmbedding(self._wordsFilePath, wordDict)
            print "saving embeddings..."
            #self._saveEmbeddingDump(self._wordsFilePath+'.preprocessed.npy',
            #                        self._wordsFilePath+'.preprocessed.dict',
            #                        self.wordEmbeddings, self.wordDict, self.wordEmbeddingsSz)

        print "loading concept embeddings..."
        if os.path.isfile(self._conceptsFilePath+'.preprocessed.dict') \
                and os.path.isfile(self._conceptsFilePath+'.preprocessed.npy'):
            self.conceptEmbeddings, self.conceptDict, self.conceptEmbeddingsSz = \
                self._loadEmbeddingDump(self._conceptsFilePath+'.preprocessed.npy', self._conceptsFilePath+'.preprocessed.dict')
        else:
            self.conceptEmbeddings, self.conceptDict, self.conceptEmbeddingsSz = \
                self._loadEmbedding(self._conceptsFilePath, conceptDict, int_key=True)
            print "saving embeddings..."
            #self._saveEmbeddingDump(self._conceptsFilePath+'.preprocessed.npy',
            #                        self._conceptsFilePath+'.preprocessed.dict',
            #                        self.conceptEmbeddings, self.conceptDict, self.conceptEmbeddingsSz)


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
