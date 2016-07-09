import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

class VanillaNNPairwiseModel:
    """
    This model is a baseline NN with simple architecture. It has an prediction
    and train function and is part of the knockoutmodel system.
    the input of the model is assumed to be 3 embedding vectors with shape (300,) each
    for the mention, the candidate sense and the context embedding sense . the output should be a simple (2,)
    binary vector, specifying which candidate is more accurate
    """

    def __init__(self, w2v):
        self._w2v = w2v
        self._batchX = []
        self._batchY = []

        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons
        self._batch_size = 32
        # self._nb_epoch = 1e3 # FIXME
        self.model = Sequential()
        self.model.add(Dense( 64 ,input_dim = w2v.embeddingSize , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense( 64 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2, init = 'uniform'))
        self.model.add(Activation('softmax'))

        # defining solver and compile
        sgd = SGD(lr=0.1, decay=1e-6,momentum=0.9)
        self.model.compile(loss='binary_crossentropy',optimizer='sgd')

    def train(self, wikilink, candidate1, candidate2, correct):
        """
        Takes a single example to train
        :param wikilink:    The wikilink to train on
        :param candidate1:  the first candidate
        :param candidate2:  the second candidate
        :param correct:     which of the two is correct (expected output)
        """

        if candidate1 not in self._w2v.conceptEmbeddings and candidate2 not in self._w2v.conceptEmbeddings:
            return None
        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        if candidate1 not in self._w2v.conceptEmbeddings:
            return candidate2
        if candidate2 not in self._w2v.conceptEmbeddings:
            return candidate1

        candidate1_vec = self._w2v.conceptEmbeddings[candidate1.lower()]
        candidate2_vec = self._w2v.conceptEmbeddings[candidate2.lower()]
        context_vec = self._w2v.meanOfWordList(wikilink['right_context'] + wikilink['left_context'])

        X = (np.asarray([context_vec, candidate1_vec, candidate2_vec])).flatten()
        Y = np.array([1,0] if candidate1 == correct else [0,1])
        self._trainXY(X,Y)

    def _trainXY(self,X,Y):
        print 'train xy'
        self._batchX.append(X)
        self._batchY.append(Y)

        if len(self._batchX) >= self._batch_size:
            print 'train first batch'
            # pushes numeric data into batch vector
            batchX = np.array(self._batchX)
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            self.model.train_on_batch(batchX, batchY)

            self._batchX = []
            self._batchY = []

    def saveModel(self, fname):
        return

    def loadModel(self, fname):
        return

    def startTraining(self):
        return
    def finilizeTraining(self):
        return

    def predict(self, wikilink, candidate1, candidate2):
        return None
