import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.layers import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

class RNNPairwiseModel:
    """
    This model is an enhancement of the SimpleNNN model that uses an 2 LSTMs
    to model the lelf context and the right context
    """

    def __init__(self, w2v, stats, context_window_sz = 20, dropout = 0.0,
                 noise=None, stripStropWords=True, stochasticContextTrimming=False):
        self._stopwords = stopwords.words('english') if stripStropWords else None
        self._w2v = w2v
        self._stats = stats
        self._batch_left_X = []
        self._batch_right_X = []
        self._batch_mention_X = []
        self._batch_candidate1_X = []
        self._batch_candidate2_X = []
        self._batch_extra_features_X = []
        self._batchY = []
        self._context_window_sz = context_window_sz
        self._train_loss = []
        self._extraFeatures = 4
        self._stochasticContextTrimming = stochasticContextTrimming

        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons
        self._batch_size = 512

        left_context_input = Input(shape=(self._context_window_sz,self._w2v.wordEmbeddingsSz), name='left_context_input')
        right_context_input = Input(shape=(self._context_window_sz,self._w2v.wordEmbeddingsSz), name='right_context_input')
        mention_input = Input(shape=(self._w2v.conceptEmbeddingsSz,), name='mention_input')
        candidate1_input = Input(shape=(self._w2v.conceptEmbeddingsSz,), name='candidate1_input')
        candidate2_input = Input(shape=(self._w2v.conceptEmbeddingsSz,), name='candidate2_input')
        extra_features_input = Input(shape=(self._extraFeatures,), name='extra_features_input')

        if noise is not None:
            left_context_input_n = GaussianNoise(noise)(left_context_input)
            right_context_input_n = GaussianNoise(noise)(right_context_input)
            mention_input_n = GaussianNoise(noise)(mention_input)
            candidate1_input_n = GaussianNoise(noise)(candidate1_input)
            candidate2_input_n = GaussianNoise(noise)(candidate2_input)
            extra_features_input_n = GaussianNoise(noise)(extra_features_input)
        else:
            left_context_input_n = left_context_input
            right_context_input_n = right_context_input
            mention_input_n = mention_input
            candidate1_input_n = candidate1_input
            candidate2_input_n = candidate2_input
            extra_features_input_n = extra_features_input

        left_lstm = GRU(self._w2v.wordEmbeddingsSz, activation='relu', dropout_U=dropout, dropout_W=dropout)(left_context_input_n)
        right_lstm = GRU(self._w2v.wordEmbeddingsSz, activation='relu', dropout_U=dropout, dropout_W=dropout)(right_context_input_n)

        x = merge([left_lstm, right_lstm, mention_input_n, candidate1_input_n, candidate2_input_n, extra_features_input_n], mode='concat')
        x = Dense(300, activation='relu')(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = Dense(50, activation='relu')(x)
        out = Dense(2, activation='softmax', name='main_output')(x)

        model = Model(input=[left_context_input, right_context_input, mention_input, candidate1_input_n, candidate2_input_n, extra_features_input_n], output=[out])
        model.compile(optimizer='adagrad', loss='binary_crossentropy')
        self.model = model

    def _context2vec(self, ctx, reverse=False):
        context_ar = self.wordListToVectors(ctx) if ctx is not None else []
        if reverse:
            context_ar = context_ar[::-1]
        if len(context_ar) >= self._context_window_sz:
            context = np.array(context_ar[-self._context_window_sz:,:])
        else:
            context = np.zeros((self._context_window_sz,self._w2v.wordEmbeddingsSz))
            if len(context_ar) != 0:
                context[-len(context_ar):,] = np.array(context_ar)
        return context

    def _2vec(self, wikilink, candidate1, candidate2):
        """
        Transforms input to w2v vectors
        returns a tuple: (wikilink vec, candidate1 vec, candidate2 vec)

        if cannot produce wikilink vec or vectors for both candidates then returns None
        if cannot produce vector to only one of the candidates then returns the id of the other
        """
        if candidate1 not in self._w2v.conceptDict and candidate2 not in self._w2v.conceptDict:
            return None
        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        if candidate1 not in self._w2v.conceptDict:
            return candidate2
        if candidate2 not in self._w2v.conceptDict:
            return candidate1

        candidate1_vec = self._w2v.conceptEmbeddings[self._w2v.conceptDict[candidate1]]
        candidate2_vec = self._w2v.conceptEmbeddings[self._w2v.conceptDict[candidate2]]
        extraFeatures_vec = np.array(self.getExtraFeatures(wikilink, candidate1, candidate2))

        left_context = self._context2vec(wikilink['left_context'] if 'left_context' in wikilink else [], False)
        right_context = self._context2vec(wikilink['right_context'] if 'right_context' in wikilink else [], True)

        mention_ar = self.wordListToVectors(wikilink['mention_as_list']) if 'mention_as_list' in wikilink else []
        mention_vec = np.mean(mention_ar, axis=0) if mention_ar.shape[0] > 0 else np.zeros(self._w2v.wordEmbeddingsSz)

        return left_context, right_context, mention_vec, candidate1_vec, candidate2_vec, extraFeatures_vec

    def wordListToVectors(self, l):
        o = []
        for w in l:
            if w in self._w2v.wordDict and (self._stopwords is None or w not in self._stopwords):
                o.append(self._w2v.wordEmbeddings[self._w2v.wordDict[w]])
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

        (left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X) = vecs
        Y = np.array([1,0] if candidate1 == correct else [0,1])
        # Check for nan
        if np.isnan(np.sum(left_X)) or np.isnan(np.sum(right_X)) \
                or np.isnan(np.sum(candidate1_X)) or np.isnan(np.sum(candidate2_X)) \
                or np.isnan(np.sum(extra_features_X)) or np.isnan(np.sum(mention_X)):
            print "Input has NaN, ignoring..."
            return
        self._trainXY(left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X, Y)

    def _trainXY(self, left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X, Y):
        self._batch_left_X.append(left_X)
        self._batch_right_X.append(right_X)
        self._batch_mention_X.append(mention_X)
        self._batch_candidate1_X.append(candidate1_X)
        self._batch_candidate2_X.append(candidate2_X)
        self._batch_extra_features_X.append(extra_features_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            # pushes numeric data into batch vector
            batch_left_X = np.array(self._batch_left_X)
            batch_right_X = np.array(self._batch_right_X)
            batch_mention_X = np.array(self._batch_mention_X)
            batch_candidate1_X = np.array(self._batch_candidate1_X)
            batch_candidate2_X = np.array(self._batch_candidate2_X)
            batch_extra_features_X = np.array(self._batch_extra_features_X)
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            loss = self.model.train_on_batch({'left_context_input':batch_left_X,
                                              'right_context_input':batch_right_X,
                                              'mention_input':batch_mention_X,
                                              'candidate1_input':batch_candidate1_X,
                                              'candidate2_input': batch_candidate2_X,
                                              'extra_features_input': batch_extra_features_X},
                                             batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss
            # print self.model.metrics_names

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_mention_X = []
            self._batch_candidate1_X = []
            self._batch_candidate2_X = []
            self._batch_extra_features_X = []
            self._batchY = []

    def plotTrainLoss(self,pairwise_model,st=0):
        plt.plot(self._train_loss[st:])
        plt.plot(pairwise_model._train_loss[10:])
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

    def getExtraFeatures(self, wlink, candidate1, candidate2):
        candidates = self._stats.getCandidatesForMention(wlink["word"])
        candidate1_prior = self._stats.getConceptPrior(candidate1)
        candidate2_prior = self._stats.getConceptPrior(candidate2)
        candidate1_conditional_prior = candidates[candidate1] if candidate1 in candidates else 0
        candidate2_conditional_prior = candidates[candidate2] if candidate2 in candidates else 0
        return [candidate1_prior, candidate2_prior, candidate1_conditional_prior, candidate2_conditional_prior]

    def predict(self, wikilink, candidate1, candidate2):
        vecs = self._2vec(wikilink, candidate1, candidate2)
        if not isinstance(vecs, tuple):
            return vecs
        (left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X) = vecs
        left_X = left_X.reshape(1,left_X.shape[0],left_X.shape[1])
        right_X = right_X.reshape(1,right_X.shape[0],right_X.shape[1])
        mention_X = mention_X.reshape(1,mention_X.shape[0])
        candidate1_X = candidate1_X.reshape(1,candidate1_X.shape[0])
        candidate2_X = candidate2_X.reshape(1,candidate2_X.shape[0])
        extra_features_X = extra_features_X.reshape(1,extra_features_X.shape[0])
        Y = self.model.predict({'left_context_input': left_X,
                                'right_context_input': right_X,
                                'mention_input': mention_X,
                                'candidate1_input': candidate1_X,
                                'candidate2_input': candidate2_X,
                                'extra_features_input': extra_features_X},
                               batch_size=1)
        return candidate1 if Y[0][0] > Y[0][1] else candidate2
