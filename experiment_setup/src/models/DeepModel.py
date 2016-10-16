import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.layers import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import theano as T
import keras.backend as K
from keras.engine.topology import Layer
from FeatureGenerator import *
import json
from PointwisePredict import *
from PairwisePredict import *

# nonzero_mean is mask_aware_mean taken from: https://github.com/fchollet/keras/issues/1579
def nonzero_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean

class NoFinetuneModelBuilder:
    def __init__(self, json, w2v):
        self._config = json
        self._w2v = w2v

        self.inputs = []
        self.to_join = []

    def addCandidateInput(self, name):
        candidate_input = Input(shape=(self._w2v.conceptEmbeddingsSz,), name=name)
        self.inputs.append(candidate_input)
        self.to_join.append(candidate_input)

    def addContextInput(self):
        left_context_input = Input(shape=(self._config['context_window_size'], self._w2v.wordEmbeddingsSz),
                                   name='left_context_input')
        right_context_input = Input(shape=(self._config['context_window_size'], self._w2v.wordEmbeddingsSz),
                                    name='right_context_input')
        left_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu')(left_context_input)
        right_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu')(right_context_input)
        self.inputs += [left_context_input, right_context_input]
        self.to_join += [left_rnn, right_rnn]

    def addMentionInput(self):
        mention_input = Input(shape=(self._w2v.conceptEmbeddingsSz,), name='mention_input')
        self.inputs.append(mention_input)
        self.to_join.append(mention_input)


class FinetuneModelBuilder:
    def __init__(self, json, w2v):
        self._config = json
        self._w2v = w2v

        self.word_embed_layer = Embedding(self._w2v.wordEmbeddings.shape[0],
                                          self._w2v.wordEmbeddingsSz,
                                          input_length=self._config['context_window_size'],
                                          weights=[self._w2v.wordEmbeddings])
#                                          init=lambda shape, name=None: K.variable(self._w2v.wordEmbeddings, name=name))
        self.concept_embed_layer = Embedding(self._w2v.conceptEmbeddings.shape[0],
                                             self._w2v.conceptEmbeddingsSz,
                                             input_length=1,
                                             weights=[self._w2v.conceptEmbeddings])
#                                             init=lambda shape, name=None: K.variable(self._w2v.conceptEmbeddings,
#                                                                                      name=name))
        self.inputs = []
        self.to_join = []

    def addCandidateInput(self, name):
        candidate_input = Input(shape=(1,), dtype='int32', name=name)
        candidate_embed = self.concept_embed_layer(candidate_input)
        candidate_flat = Flatten()(candidate_embed)
        self.inputs.append(candidate_input)
        self.to_join.append(candidate_flat)

    def addContextInput(self):
        left_context_input = Input(shape=(self._config['context_window_size'],), dtype='int32', name='left_context_input')
        right_context_input = Input(shape=(self._config['context_window_size'],), dtype='int32', name='right_context_input')
        left_context_embed = self.word_embed_layer(left_context_input)
        right_context_embed = self.word_embed_layer(right_context_input)
        left_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu')(left_context_embed)
        right_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu')(right_context_embed)
        self.inputs += [left_context_input, right_context_input]
        self.to_join += [left_rnn, right_rnn]

    def addMentionInput(self):
        mention_input = Input(shape=(self._config['max_mention_words'],), dtype='int32', name='mention_input')
        mention_embed = self.word_embed_layer(mention_input)
        mention_mean = Lambda(nonzero_mean, output_shape=(self._w2v.conceptEmbeddingsSz,))(mention_embed)
        self.inputs.append(mention_input)
        self.to_join.append(mention_mean)


class DeepModel:
    def __init__(self, config, load_path=None, w2v=None, db=None, stats=None):
        '''
        Creates a new NN model configured by a json.

        config is either a dict or a path to a json file

        json structure:
        {
            strip_stop_words=[boolean]
            context_window_size=[int]
            max_mention_words=[int]
            dropout=[0.0 .. 1.0]
            feature_generator={mention_features={feature names...}, entity_features={feature names...}}

            finetune_embd=[boolean]
            pairwise=[boolean]
            inputs=[list out of ['candidates', 'context', 'mention', 'extra_features']]
        }
        '''

        if type(config) == str:
            with open(config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config
        self._stopwords = stopwords.words('english') if self._config['strip_stop_words'] else None
        self._w2v = w2v
        self._batch_left_X = []
        self._batch_right_X = []
        self._batch_candidate1_X = []
        self._batch_candidate2_X = []
        self._batch_mention_X = []
        self._batch_extra_features_X = []
        self._batchY = []
        self._train_loss = []
        self._batch_size = 512
        self.inputs = {x for x in self._config['inputs']}

        if 'feature_generator' in self._config:
            self._feature_generator = FeatureGenerator(mention_features=
                                                       self._config['feature_generator']['mention_features'],
                                                       entity_features=
                                                       self._config['feature_generator']['entity_features'],
                                                       stats=stats, db=db)
        self.model = None

        if load_path is None:
            self.compileModel(dropout=self._config['dropout'])
        else:
            self.loadModel(load_path)

    def getPredictor(self):
        if self._config['pairwise']:
            return PairwisePredict(self)
        else:
            return PointwisePredict(self)

    def compileModel(self, dropout=0.0):
        model_builder = FinetuneModelBuilder(self._config, self._w2v) if self._config['finetune_embd'] \
            else NoFinetuneModelBuilder(self._config, self._w2v)

        if 'candidates' in self.inputs:
            model_builder.addCandidateInput('candidate1_input')
            if self._config['pairwise']:
                model_builder.addCandidateInput('candidate2_input')

        if 'context' in self.inputs:
            model_builder.addContextInput()

        if 'mention' in self.inputs:
            model_builder.addMentionInput()

        inputs = model_builder.inputs
        to_join = model_builder.to_join

        if 'extra_features' in self.inputs:
            n_extra_features = self._feature_generator.numPairwiseFeatures() if self._config['pairwise'] \
                else self._feature_generator.numPointwiseFeatures()
            extra_features_input = Input(shape=(n_extra_features,), name='extra_features_input')
            inputs.append(extra_features_input)
            to_join.append(extra_features_input)

        # join all inputs
        x = merge(to_join, mode='concat') if len(to_join) > 1 else to_join[0]

        # build classifier model
        x = Dense(300, activation='relu')(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        out = Dense(2, activation='softmax', name='main_output')(x)

        model = Model(input=inputs, output=[out])
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
        if (candidate1 is None or candidate1 not in self._w2v.conceptDict) and \
                (candidate2 is None or candidate2 not in self._w2v.conceptDict):
            return None
        if self._config['pairwise']:
            if candidate1 is None or candidate1 not in self._w2v.conceptDict:
                return candidate2
            if candidate2 is None or candidate2 not in self._w2v.conceptDict:
                return candidate1

        candidate1_X = None
        candidate2_X = None
        left_context_X = None
        right_context_X = None
        mention_X = None
        extra_features_X = None

        # get candidate inputs
        if 'candidates' in self.inputs:
            if self._config['finetune_embd']:
                candidate1_X = np.array([self._w2v.conceptDict[candidate1]]) if candidate1 is not None else None
                candidate2_X = np.array([self._w2v.conceptDict[candidate2]]) if candidate2 is not None else None
            else:
                candidate1_X = self._w2v.conceptEmbeddings[self._w2v.conceptDict[candidate1]] if candidate1 is not None \
                    else None
                candidate2_X = self._w2v.conceptEmbeddings[self._w2v.conceptDict[candidate2]] if candidate2 is not None \
                    else None

        # get context input
        if 'context' in self.inputs:
            if self._config['finetune_embd']:
                left_context_X = self.wordListToIndices(wikilink['left_context'],
                                                        self._config['context_window_size'],
                                                        reverse=False)
                right_context_X = self.wordListToIndices(wikilink['right_context'],
                                                         self._config['context_window_size'],
                                                         reverse=True)
            else:
                left_context_X = self.wordListToVectors(wikilink['left_context'],
                                                        self._config['context_window_size'],
                                                        reverse=False)
                right_context_X = self.wordListToVectors(wikilink['right_context'],
                                                         self._config['context_window_size'],
                                                         reverse=True)

        # get mention input
        if 'mention' in self.inputs:
            if self._config['finetune_embd']:
                mention_X = self.wordListToIndices(wikilink['mention_as_list'],
                                                   self._config['max_mention_words'],
                                                   reverse=False)
            else:
                mention_ar = self.wordListToVectors(wikilink['mention_as_list'],
                                                    self._config['max_mention_words'],
                                                    reverse=False) if 'mention_as_list' in wikilink else []
                mention_X = np.mean(mention_ar, axis=0) if mention_ar.shape[0] > 0 else np.zeros(
                    self._w2v.wordEmbeddingsSz)

        if 'extra_features' in self.inputs:
            if self._config['pairwise']:
                extra_features_X = \
                    np.array(self._feature_generator.getPairwiseFeatures(wikilink, candidate1, candidate2))
            else:
                extra_features_X = \
                    np.array(self._feature_generator.getPointwiseFeatures(wikilink, candidate1))

        return left_context_X, right_context_X, mention_X, candidate1_X, candidate2_X, extra_features_X


    def wordListToIndices(self, l, output_len, reverse):
        o = []
        for w in l:
            if w in self._w2v.wordDict and (self._stopwords is None or w not in self._stopwords):
                o.append(self._w2v.wordDict[w])
        if len(o) == 0:
            o.append(self._w2v.wordDict[self._w2v.DUMMY_KEY])
        if reverse:
            o = o[::-1]
        arr = np.zeros((output_len,))
        n = len(o) if len(o) <= output_len else output_len
        arr[:n] = np.array(o)[:n]
        return arr

    def wordListToVectors(self, l, output_len, reverse):
        o = []
        for w in l:
            if w in self._w2v.wordDict and (self._stopwords is None or w not in self._stopwords):
                o.append(self._w2v.wordEmbeddings[self._w2v.wordDict[w]])
        o = np.asarray(o)

        if reverse:
            o = o[::-1]
        if len(o) >= output_len:
            context = np.array(o[-self._config['context_window_size']:, :])
        else:
            context = np.zeros((self._config['context_window_size'], self._w2v.wordEmbeddingsSz))
            if len(o) != 0:
                context[-len(o):, ] = np.array(o)
        return context

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

    def _trainXY(self, left_X, right_X, mention_X, candidate1_X, candidate2_X, extra_features_X, Y):
        self._batch_left_X.append(left_X)
        self._batch_right_X.append(right_X)
        self._batch_mention_X.append(mention_X)
        self._batch_candidate1_X.append(candidate1_X)
        self._batch_candidate2_X.append(candidate2_X)
        self._batch_extra_features_X.append(extra_features_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            batchX = {}
            if 'candidates' in self.inputs:
                batchX['candidate1_input'] = np.array(self._batch_candidate1_X)
                if self._config['pairwise']:
                    batchX['candidate2_input'] = np.array(self._batch_candidate2_X)
            if 'context' in self.inputs:
                batchX['left_context_input'] = np.array(self._batch_left_X)
                batchX['right_context_input'] = np.array(self._batch_right_X)
            if 'mention' in self.inputs:
                batchX['mention_input'] = np.array(self._batch_mention_X)
            if 'extra_features' in self.inputs:
                batchX['extra_features_input'] = np.array(self._batch_extra_features_X)
            batchY = np.array(self._batchY)

            #for x,y in batchX.iteritems():
            #    print x, ":", y.shape
            #print "Y:", batchY.shape

            loss = self.model.train_on_batch(batchX, batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_mention_X = []
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

    def predict(self, wikilink, candidate1, candidate2):
        vecs = self._2vec(wikilink, candidate1, candidate2)
        if not isinstance(vecs, tuple):
            return vecs
        (left_X, right_X, mention_X, candidate1_X, candidate2_X, extraFeatures_X) = vecs

        X = {}
        if 'candidates' in self.inputs:
            X['candidate1_input'] = candidate1_X.reshape((1, candidate1_X.shape[0],))
            if self._config['pairwise']:
                X['candidate2_input'] = candidate2_X.reshape((1, candidate2_X.shape[0],))
        if 'context' in self.inputs:
            if self._config['finetune_embd']:
                X['left_context_input'] = left_X.reshape((1, left_X.shape[0],))
                X['right_context_input'] = right_X.reshape((1, right_X.shape[0],))
            else:
                X['left_context_input'] = left_X.reshape(1, left_X.shape[0], left_X.shape[1])
                X['right_context_input'] = right_X.reshape(1, right_X.shape[0], right_X.shape[1])
        if 'mention' in self.inputs:
            X['mention_input'] = mention_X.reshape((1, mention_X.shape[0],))
        if 'extra_features' in self.inputs:
            X['extra_features_input'] = np.array(extraFeatures_X.reshape(1,extraFeatures_X.shape[0]))

        Y = self.model.predict(X, batch_size=1)
        return Y[0][0]