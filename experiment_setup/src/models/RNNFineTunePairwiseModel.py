import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.layers import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import theano as T

class NonZeroAverage(Layer):
    """
    Layer that averages over non-zero word embeddings to produce an average vector for e.g. an entity or an aspect.
    Not fully implemented yet.
    """
    @property
    def output_shape(self):
        shape = list(self.input_shape)
        assert len(shape) == 3 # only valid for 3D tensors
        return tuple([shape[0], shape[2]])

    def get_output(self, train=False):
        x = self.get_input(train)
        shape = list(self.input_shape)
        sums = x.sum(axis=1)
        c = T.neq(x,0).sum(axis=2)
        count = T.neq(c,0).sum(axis=1)
        t = [count] * shape[2]
        stacked = T.stack(*t).transpose()
        ave = sums / stacked
        return ave.astype('float32')

class RNNFineTunePairwiseModel:
    """
    This model is an enhancement of the SimpleNNN model that uses an 2 LSTMs
    to model the lelf context and the right context


    !!! OUT OF DATE - Need to fit to new RNNPairwiseModel !!!



    """

    def __init__(self, w2v, context_window_sz=10, dropout=0.0, noise=None, stripStropWords=True,
                 feature_generator=None, max_mention_words=5):
        self._stopwords = stopwords.words('english') if stripStropWords else None
        self._w2v = w2v
        self._batch_left_X = []
        self._batch_right_X = []
        self._batch_candidate1_X = []
        self._batch_candidate2_X = []
        self._batch_mention_X = [[] for i in xrange(max_mention_words)]
        self._batch_extra_features_X = []
        self._batchY = []
        self._context_window_sz = context_window_sz
        self._train_loss = []
        self._batch_size = 512
        self._max_mention_words = max_mention_words
        self._feature_generator = feature_generator
        self.model = None
        self.compileModel(dropout=dropout, noise=noise)

    def compileModel(self, dropout=0.0, noise=None):
        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons

#        word_embed_layer = Embedding(self._w2v.wordEmbeddings.shape[0],
#                                     self._w2v.wordEmbeddingsSz,
#                                     weights=[self._w2v.wordEmbeddings])
#        concept_embed_layer = Embedding(self._w2v.conceptEmbeddings.shape[0],
#                                        self._w2v.conceptEmbeddingsSz,
#                                        input_length=1,
#                                        weights=[self._w2v.conceptEmbeddings])
        word_embed_layer = Embedding(self._w2v.wordEmbeddings.shape[0],
                                     self._w2v.wordEmbeddingsSz,
                                     init=lambda shape, name=None: K.variable(self._w2v.wordEmbeddings, name=name))
        concept_embed_layer = Embedding(self._w2v.conceptEmbeddings.shape[0],
                                        self._w2v.conceptEmbeddingsSz,
                                         input_length=1,
                                        init=lambda shape, name=None: K.variable(self._w2v.conceptEmbeddings, name=name))

        def wordEmbeddingsInit(shape, name=None):
            value = np.random.random(shape)
            return K.variable(value, name=name)

        left_context_input = Input(shape=(self._context_window_sz,), dtype='int32', name='left_context_input')
        right_context_input = Input(shape=(self._context_window_sz,), dtype='int32', name='right_context_input')
        candidate1_input = Input(shape=(1,), dtype='int32', name='candidate1_input')
        candidate2_input = Input(shape=(1,), dtype='int32', name='candidate2_input')
#        mention_input = Input(shape=(self._max_mention_words,), dtype='int32', name='mention_input')
#        mention_input = [Input(shape=(1,), dtype='int32', name='mention_input' + str(k))
#                         for k in xrange(self._max_mention_words)]
        if self._feature_generator is not None:
            extra_features_input = Input(shape=(self._feature_generator.numPairwiseFeatures(),), name='extra_features_input')

        left_context_embed = word_embed_layer(left_context_input)
        right_context_embed = word_embed_layer(right_context_input)
        candidate1_embed = concept_embed_layer(candidate1_input)
        candidate2_embed = concept_embed_layer(candidate2_input)
#        mention_embed = word_embed_layer(mention_input)
#        mention_embed = [word_embed_layer(i) for i in mention_input]
        if noise is not None:
            left_context_embed = GaussianNoise(noise)(left_context_embed)
            right_context_embed = GaussianNoise(noise)(right_context_embed)
            candidate1_embed = GaussianNoise(noise)(candidate1_embed)
            candidate2_embed = GaussianNoise(noise)(candidate2_embed)
#            mention_embed = GaussianNoise(noise)(mention_embed)
#            mention_embed = [GaussianNoise(noise)(i) for i in mention_embed]
            if self._feature_generator is not None:
                extra_features_input_n = GaussianNoise(noise)(extra_features_input)
        else:
            extra_features_input_n = extra_features_input

#        mention_flat = Flatten()(mention_embed)
#        mention_flat = [Flatten()(i) for i in mention_embed]
        candidate1_flat = Flatten()(candidate1_embed)
        candidate2_flat = Flatten()(candidate2_embed)

 #       mention_avg = merge(mention_flat, mode='ave')

        left_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu', return_sequences=False, dropout_U=dropout, dropout_W=dropout)(left_context_embed)
        right_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu', return_sequences=False, dropout_U=dropout, dropout_W=dropout)(right_context_embed)

#        if noise is not None:
#            x = merge([left_rnn, right_rnn, candidate1_flat, candidate2_flat, mention_avg, extra_features_input_n], mode='concat')
#        else:
#            x = merge([left_rnn, right_rnn, candidate1_flat, candidate2_flat, mention_avg], mode='concat')
        if noise is not None:
            x = merge([left_rnn, right_rnn, candidate1_flat, candidate2_flat, extra_features_input_n], mode='concat')
        else:
            x = merge([left_rnn, right_rnn, candidate1_flat, candidate2_flat], mode='concat')

        x = Dense(300, activation='relu')(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = Dense(50, activation='relu')(x)
        out = Dense(2, activation='softmax', name='main_output')(x)

        if self._feature_generator is not None:
            model = Model(input=[left_context_input, right_context_input, candidate1_input,
                                 candidate2_input, extra_features_input],
                          output=[out])
        else:
            model = Model(input=[left_context_input, right_context_input, candidate1_input,
                                 candidate2_input],
                          output=[out])
#        if noise is not None:
#            model = Model(input=[left_context_input, right_context_input, candidate1_input,
#                                 candidate2_input, extra_features_input] + mention_input,
#                          output=[out])
#        else:
#            model = Model(input=[left_context_input, right_context_input, candidate1_input,
#                                 candidate2_input] + mention_input,
#                          output=[out])
        model.compile(optimizer='adagrad', loss='binary_crossentropy')
        self.model = model
        print "model compiled!"

    def _2vec(self, wikilink, candidate1, candidate2):
        """
        Transforms input to w2v vectors
        returns a tuple: (wikilink vec, candidate1 vec, candidate2 vec)

        if cannot produce wikilink vec or vectors for both candidates then returns None
        if cannot produce vector to only one of the candidates then returns the id of the other
        """
        if candidate1 not in self._w2v.conceptDict and candidate2 not in self._w2v.conceptDict:
            return None
        if candidate1 not in self._w2v.conceptDict:
            return candidate2
        if candidate2 not in self._w2v.conceptDict:
            return candidate1

        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        candidate1_id = np.array([self._w2v.conceptDict[candidate1]])
        candidate2_id = np.array([self._w2v.conceptDict[candidate2]])

        left_context = self.wordListToIndices(wikilink['left_context'], self._context_window_sz, reverse=False)
        right_context = self.wordListToIndices(wikilink['right_context'], self._context_window_sz, reverse=True)
        mention = self.wordListToIndices(wikilink['mention_as_list'], self._max_mention_words, reverse=False)

        extraFeatures_vec = np.array(self._feature_generator.getPairwiseFeatures(wikilink, candidate1, candidate2)) \
            if self._feature_generator is not None else None

        return (left_context, right_context, mention, candidate1_id, candidate2_id, extraFeatures_vec)

    def wordListToIndices(self, l, output_len, reverse):
        o = []
        for w in l:
            if w in self._w2v.wordDict and (self._stopwords is None or w not in self._stopwords):
                o.append(self._w2v.wordDict[w])
        if reverse:
            o = o[::-1]

        arr = np.zeros((self._context_window_sz,))
        n = len(o) if len(o) <= output_len else output_len
        arr[:n] = np.array(o)[:n]
        return arr

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
        self._trainXY(left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X, Y)

    def _trainXY(self,left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X, Y):
        self._batch_left_X.append(left_X)
        self._batch_right_X.append(right_X)
        for i in xrange(self._max_mention_words):
            self._batch_mention_X[i].append(mention_X[i])
        self._batch_candidate1_X.append(candidate1_X)
        self._batch_candidate2_X.append(candidate2_X)
        self._batch_extra_features_X.append(extra_features_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            # pushes numeric data into batch vector
            batchX = {'left_context_input': np.array(self._batch_left_X),
                      'right_context_input': np.array(self._batch_right_X),
                      'candidate1_input': np.array(self._batch_candidate1_X),
                      'candidate2_input': np.array(self._batch_candidate2_X)}
            if self._feature_generator is not None:
                batchX['extra_features_input'] = np.array(self._batch_extra_features_X)
#            for i in xrange(self._max_mention_words):
#                batchX['mention_input' + str(i)] = np.array(self._batch_mention_X[i])
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            loss = self.model.train_on_batch(batchX, batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss
            # print self.model.metrics_names

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_mention_X = [[] for i in xrange(self._max_mention_words)]
            self._batch_candidate1_X = []
            self._batch_candidate2_X = []
            self._batch_extra_features_X = []
            self._batchY = []

    def plotTrainLoss(self,fname, st=0):
        plt.plot(self._train_loss[st:])
        plt.ylabel('Loss')
        plt.xlabel('Batch')
        plt.savefig(fname)

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
        (left_X, right_X, mention_X, candidate1_X, candidate2_X, extraFeatures_X) = vecs

        X = {'left_context_input': left_X.reshape((1, left_X.shape[0])),
             'right_context_input': right_X.reshape((1, right_X.shape[0],)),
             'mention_input': mention_X.reshape((1, mention_X.shape[0],)),
             'candidate1_input': candidate1_X,
             'candidate2_input': candidate2_X}
        if self._feature_generator is not None:
            X['extra_features_input'] = np.array(extraFeatures_X.reshape(1,extraFeatures_X.shape[0]))
 #       for i in xrange(self._max_mention_words):
 #           X['mention_input' + str(i)] = mention_X[i].reshape((1, mention_X[i].shape[0],))

        Y = self.model.predict(X, batch_size=1)
        return candidate1 if Y[0][0] > Y[0][1] else candidate2
