import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Model
from keras.models import model_from_json
from nltk.corpus import stopwords


class RNNPairwiseModel:
    """

    !!! OUT OF DATE - Need to fit to new RNNPairwiseModel !!!

    """

    def __init__(self, w2v, stats=None, context_window_sz = 10, dropout = 0.0, noise = None, stripStropWords=True, addPriorFeature=False):
        self._stopwords = stopwords.words('english') if stripStropWords else None
        self._w2v = w2v
        self._batch_left_X = []
        self._batch_right_X = []
        self._batch_entity_X = []
        self._batchY = []
        self._context_window_sz = context_window_sz
        self._train_loss = []
        self._stats = stats
        self._addPriorFeature = addPriorFeature
        if addPriorFeature and stats is None:
            raise Exception("If addPriorFeature is True then stat object must be supplied")

        self._batch_size = 512

        left_context_input = Input(shape=(self._context_window_sz,self._w2v.wordEmbeddingsSz), name='left_context_input')
        right_context_input = Input(shape=(self._context_window_sz,self._w2v.wordEmbeddingsSz), name='right_context_input')
        entity_input = Input(shape=(self._w2v.conceptEmbeddingsSz + 1,), name='entity_input')

        if noise is not None:
            left_context_input_n = GaussianNoise(noise)(left_context_input)
            right_context_input_n = GaussianNoise(noise)(right_context_input)
            entity_input_n = GaussianNoise(noise)(entity_input)
        else:
            left_context_input_n = left_context_input
            right_context_input_n = right_context_input
            entity_input_n = entity_input

        # bidirectional GRU model merged with the word embeddings trained with w2v
        left_gru = GRU(self._w2v.wordEmbeddingsSz, activation='relu', dropout_U=dropout, dropout_W=dropout)(left_context_input_n)
        right_gru = GRU(self._w2v.wordEmbeddingsSz, activation='relu', dropout_U=dropout, dropout_W=dropout)(right_context_input_n)
        context = merge([left_gru, right_gru], mode='sum')

        x = merge([context, entity_input_n], mode='dot')
        out = Dense(1,activation='sigmoid')(x)

        model = Model(input=[left_context_input, right_context_input, entity_input], output=[out])
        model.compile(optimizer='adagrad', loss='mean_squared_error')
        self.model = model

    def _2vec(self, wikilink, entity, entity_prior=0):
        """
        Transforms input to w2v vectors
        returns a tuple: (wikilink vec, candidate1 vec, candidate2 vec)

        if cannot produce wikilink vec or vectors for both candidates then returns None
        if cannot produce vector to only one of the candidates then returns the id of the other
        """

        if entity not in self._w2v.conceptDict:
            return None
        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        entity_vec = self._w2v.conceptEmbeddings[self._w2v.conceptDict[entity]]
        if not self._addPriorFeature:
            entity_prior = 0

        entity_vec = np.zeros(self._w2v.conceptEmbeddingsSz + 1)
        entity_vec[0:self._w2v.conceptEmbeddingsSz] = self._w2v.conceptEmbeddings[self._w2v.conceptDict[entity]]
        entity_vec[self._w2v.conceptEmbeddingsSz:self._w2v.conceptEmbeddingsSz] = entity_prior

        left_context_ar = self.wordListToVectors(wikilink['left_context']) if 'left_context' in wikilink else []
        if len(left_context_ar) >= self._context_window_sz:
            left_context = np.array(left_context_ar[-self._context_window_sz:,:])
        else:
            left_context = np.zeros((self._context_window_sz,self._w2v.wordEmbeddingsSz))
            if len(left_context_ar) != 0:
                left_context[-len(left_context_ar):,] = np.array(left_context_ar)

        if 'right_context' in wikilink:
            right_context_ar = self.wordListToVectors(wikilink['right_context'])[::-1]
        else:
            right_context_ar = []
        if len(right_context_ar) >= self._context_window_sz:
            right_context = np.array(right_context_ar[-self._context_window_sz:,:])
        else:
            right_context = np.zeros((self._context_window_sz,self._w2v.wordEmbeddingsSz))
            if len(right_context_ar) != 0:
                right_context[-len(right_context_ar):,] = np.array(right_context_ar)

        return left_context, right_context, entity_vec

    def wordListToVectors(self, l):
        o = []
        for w in l:
            if w in self._w2v.wordDict and (self._stopwords is None or w not in self._stopwords):
                o.append(self._w2v.wordEmbeddings[self._w2v.wordDict[w]])
        return np.asarray(o)

    def train(self, wikilink, entity, correct, entity_prior=0):
        """
        Takes a single example to train
        :param wikilink:    The wikilink to train on
        :param entity:      the entity
        :param correct:     correct or not (neg-sample)
        """
        vecs = self._2vec(wikilink, entity, entity_prior)
        if not isinstance(vecs, tuple):
            return # nothing to train on

        (left_X, right_X, entity_X) = vecs
        Y = np.array([1.0] if correct else [0.0])
        # Check for nan
        if np.isnan(np.sum(left_X)) or np.isnan(np.sum(right_X)) or np.isnan(np.sum(entity_X)):
            print "Input has NaN, ignoring..."
            return
        self._trainXY(left_X, right_X, entity_X,Y)

    def _trainXY(self,left_X, right_X, entity_X,Y):
        self._batch_left_X.append(left_X)
        self._batch_right_X.append(right_X)
        self._batch_entity_X.append(entity_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            # pushes numeric data into batch vector
            batch_left_X = np.array(self._batch_left_X)
            batch_right_X = np.array(self._batch_right_X)
            _batch_entity_X = np.array(self._batch_entity_X)
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            loss = self.model.train_on_batch({'left_context_input':batch_left_X,
                                              'right_context_input':batch_right_X,
                                              'entity_input':_batch_entity_X},
                                             batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss
            # print self.model.metrics_names

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_entity_X = []
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

    def predict(self, wikilink, entity, entity_prior=0):
        vecs = self._2vec(wikilink, entity, entity_prior)
        if not isinstance(vecs, tuple):
            return vecs
        (left_X, right_X, entity_X) = vecs
        left_X = left_X.reshape(1,left_X.shape[0],left_X.shape[1])
        right_X = right_X.reshape(1,right_X.shape[0],right_X.shape[1])
        entity_X = entity_X.reshape(1,entity_X.shape[0])
        Y = self.model.predict({'left_context_input':left_X,
                                'right_context_input':right_X,
                                'entity_input':entity_X},
                               batch_size=1)
        return Y[0][0]
