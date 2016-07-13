import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.layers import *
import matplotlib.pyplot as plt

class RNNPairwiseModel:
    """
    This model is an enhancement of the SimpleNNN model that uses an 2 LSTMs
    to model the lelf context and the right context
    """

    def __init__(self, w2v, context_window_sz = 10):
        self._w2v = w2v
        self._batch_left_X = []
        self._batch_right_X = []
        self._batch_candidates_X = []
        self._batchY = []
        self._context_window_sz = context_window_sz
        self._train_loss = []

        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons
        self._batch_size = 512

        left_context_input = Input(shape=(self._context_window_sz,self._w2v.embeddingSize), name='left_context_input')
        right_context_input = Input(shape=(self._context_window_sz,self._w2v.embeddingSize), name='right_context_input')
        candidates_input = Input(shape=(self._w2v.embeddingSize * 2,), name='candidates_input')

        left_lstm = GRU(self._w2v.embeddingSize, activation='relu')(left_context_input)
        right_lstm = GRU(self._w2v.embeddingSize, activation='relu')(right_context_input)

        x = merge([left_lstm, right_lstm,candidates_input], mode='concat')
        x = Dense(300, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        out = Dense(2, activation='softmax', name='main_output')(x)

        model = Model(input=[left_context_input, right_context_input,candidates_input], output=[out])
        model.compile(optimizer='adagrad', loss='binary_crossentropy')
        self.model = model

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
        candidates = (np.asarray([candidate1_vec, candidate2_vec])).flatten()


        left_context_ar = self.wordListToVectors(wikilink['left_context'])
        if (len(left_context_ar) >= self._context_window_sz):
            left_context = np.array(left_context_ar[-self._context_window_sz:,:])
        else:
            left_context = np.zeros((self._context_window_sz,self._w2v.embeddingSize))
            if len(left_context_ar) != 0:
                left_context[-len(left_context_ar):,] = np.array(left_context_ar)

        right_context_ar = self.wordListToVectors(wikilink['right_context'])[::-1]
        if (len(right_context_ar) >= self._context_window_sz):
            right_context = np.array(right_context_ar[-self._context_window_sz:,:])
        else:
            right_context = np.zeros((self._context_window_sz,self._w2v.embeddingSize))
            if len(right_context_ar) != 0:
                right_context[-len(right_context_ar):,] = np.array(right_context_ar)

        return (left_context, right_context,candidates)

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

        (left_X, right_X, candidates_X) = vecs
        Y = np.array([1,0] if candidate1 == correct else [0,1])
        # Check for nan
        if np.isnan(np.sum(left_X)) or np.isnan(np.sum(right_X)) or np.isnan(np.sum(candidates_X)):
            print "Input has NaN, ignoring..."
            return
        self._trainXY(left_X, right_X, candidates_X,Y)

    def _trainXY(self,left_X, right_X, candidates_X,Y):
        self._batch_left_X.append(left_X)
        self._batch_right_X.append(right_X)
        self._batch_candidates_X.append(candidates_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            # pushes numeric data into batch vector
            batch_left_X = np.array(self._batch_left_X)
            batch_right_X = np.array(self._batch_right_X)
            batch_candidates_X = np.array(self._batch_candidates_X)
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            loss = self.model.train_on_batch({'left_context_input':batch_left_X,
                                              'right_context_input':batch_right_X,
                                              'candidates_input':batch_candidates_X},
                                             batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss
            # print self.model.metrics_names

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_candidates_X = []
            self._batchY = []

    def plotTrainLoss(self,st=0):
        plt.plot(self._train_loss[st:])
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
        (left_X, right_X, candidates_X) = vecs
        left_X = left_X.reshape(1,left_X.shape[0],left_X.shape[1])
        right_X = right_X.reshape(1,right_X.shape[0],right_X.shape[1])
        candidates_X = candidates_X.reshape(1,candidates_X.shape[0])
        Y = self.model.predict({'left_context_input':left_X,
                                'right_context_input':right_X,
                                'candidates_input':candidates_X},
                               batch_size=1)
        return candidate1 if Y[0][0] > Y[0][1] else candidate2
