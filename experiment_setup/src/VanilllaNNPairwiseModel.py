import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import *
import matplotlib.pyplot as plt

class VanillaNNPairwiseModel:
    """
    This model is a baseline NN with simple architecture. It has an prediction
    and train function and is part of the knockoutmodel system.
    the input of the model is assumed to be 3 embedding vectors with shape (300,) each
    for the mention, the candidate sense and the context embedding sense . the output should be a simple (2,)
    binary vector, specifying which candidate is more accurate
    """

    def __init__(self, w2v, context_window_sz = 20, lstm=None):
        self._w2v = w2v
        self._batchX = []
        self._batchY = []
        self._context_window_sz = context_window_sz
        self._train_loss = []
        self._lstm = lstm

        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons
        self._batch_size = 4096
        # self._nb_epoch = 1e3 # FIXME
        self.model = Sequential()
        self.model.add(Dense( 300 ,input_dim = w2v.embeddingSize * 3 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense( 50 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2, init = 'uniform'))
        self.model.add(Activation('softmax'))


        # defining solver and compile
        self.model.compile(loss='binary_crossentropy',optimizer='adagrad')

    def _2vec(self, wikilink, candidate1, candidate2):
        """
        Transforms input to w2v vectors
        returns a tuple: (wikilink vec, candidate1 vec, candidate2 vec)

        if cannot produce wikilink vec or vectors for both candidates then returns None
        if cannot produce vector to only one of the candidates then returns the id of the other
        """
        if candidate1 not in self._w2v.conceptEmbeddings and candidate2 not in self._w2v.conceptEmbeddings:
            return None
        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        if candidate1 not in self._w2v.conceptEmbeddings:
            return candidate2
        if candidate2 not in self._w2v.conceptEmbeddings:
            return candidate1

        candidate1_vec = self._w2v.conceptEmbeddings[candidate1]
        candidate2_vec = self._w2v.conceptEmbeddings[candidate2]

        if self._lstm is None:
            # average
            context = []
            if len(wikilink['right_context']) <= self._context_window_sz:
                context += wikilink['right_context']
            else:
                context += wikilink['right_context'][-self._context_window_sz:]
            if len(wikilink['left_context']) <= self._context_window_sz:
                context += wikilink['left_context']
            else:
                context += wikilink['left_context'][-self._context_window_sz:]

            context_vec = self._w2v.meanOfWordList(context)
        else:
            # use lstm
            ar = self.wordListToVectors(wikilink['left_context'])
            if ar.shape[0] < self._context_window_sz:
                return None

            # cache the context embedding. We are likely to see the same wikilink a number of times
            if '_context_embed' not in wikilink:
                context_vec = self._lstm.predict(np.array([ar[-10:,:]]), batch_size=1)
                context_vec = context_vec.flatten()
                wikilink['_context_embed'] = context_vec
            else:
                context_vec = wikilink['_context_embed']

        return (context_vec, candidate1_vec, candidate2_vec)

    def wordListToVectors(self, l):
        o = []
        for w in l:
            if w in self._w2v.wordEmbeddings:
                o.append(self._w2v.wordEmbeddings[w])
        return np.asarray(o)

    def train(self, wikilink, candidate1, candidate2, correct):
        """
        Takes a single example to train
        :param wikilink:    The wikilink to train on
        :param candidate1:  the first candidate
        :param candidate2:  the second candidate
        :param correct:     which of the two is correct (expected output)
        """
        vecs = self._2vec(wikilink, candidate1, candidate2)
        if not isinstance(vecs, tuple):
            return # nothing to train on
        (context_vec, candidate1_vec, candidate2_vec) = vecs

        X = (np.asarray([context_vec, candidate1_vec, candidate2_vec])).flatten()
        Y = np.array([1,0] if candidate1 == correct else [0,1])
        # Check for nan
        if np.isnan(np.sum(X)):
            print "Input has NaN, ignoring..."
            return
        self._trainXY(X,Y)

    def _trainXY(self,X,Y):
        self._batchX.append(X)
        self._batchY.append(Y)

        if len(self._batchX) >= self._batch_size:
            # pushes numeric data into batch vector
            batchX = np.array(self._batchX)
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            loss = self.model.train_on_batch(batchX, batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch x - ', batchX.shape, '; loss: ', loss
            # print self.model.metrics_names

            self._batchX = []
            self._batchY = []

    def plotTrainLoss(self):
        plt.plot(self._train_loss)
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.show()

    def saveModel(self, fname):
        open(fname+".model", 'w').write(self.model.to_json())
        self.model.save_weights(fname + ".weights", overwrite=True)
        return

    def loadModel(self, fname):
        self.model = model_from_json(open(fname+".model", 'r').read())
        self.model.load_weights(fname + ".weights")
        return

    def startTraining(self):
        return

    def finilizeTraining(self):
        return

    def predict(self, wikilink, candidate1, candidate2):
        vecs = self._2vec(wikilink, candidate1, candidate2)
        if not isinstance(vecs, tuple):
            return vecs
        (context_vec, candidate1_vec, candidate2_vec) = vecs

        X = np.asarray([context_vec, candidate1_vec, candidate2_vec])
        X = X.reshape(1,X.size)
        Y = self.model.predict(X, batch_size=1)
        return candidate1 if Y[0][0] > Y[0][1] else candidate2
