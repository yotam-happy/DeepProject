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

def sum_seq(x):
    return K.sum(x, axis=1, keepdims=False)

def to_prob(input):
    sum = K.sum(input, 1, keepdims=True)
    return input / sum

def get_activations1(model, layer, X_batch):
    if layer == -1:
        return [X_batch]
    else:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
        activations = get_activations([X_batch, 0])
        return activations


class ModelBuilder:
    def __init__(self, config_json, w2v):
        self._config = config_json
        self._w2v = w2v

        self.word_embed_layer = Embedding(self._w2v.wordEmbeddings.shape[0],
                                          self._w2v.wordEmbeddingsSz,
                                          input_length=self._config['context_window_size'],
                                          weights=[self._w2v.wordEmbeddings],
                                          trainable=self._config['finetune_embd'])
        self.concept_embed_layer = Embedding(self._w2v.conceptEmbeddings.shape[0],
                                             self._w2v.conceptEmbeddingsSz,
                                             input_length=1,
                                             weights=[self._w2v.conceptEmbeddings],
                                             trainable=self._config['finetune_embd'])
        self.inputs = []
        self.to_join = []

    def addCandidateInput(self, name, to_join=True):
        candidate_input = Input(shape=(1,), dtype='int32', name=name)
        candidate_embed = self.concept_embed_layer(candidate_input)
        candidate_flat = Flatten()(candidate_embed)
        self.inputs.append(candidate_input)
        if to_join:
            self.to_join.append(candidate_flat)
        return candidate_flat

    def buildAttention(self, seq, controller):
        controller_repeated = RepeatVector(self._config['context_window_size'])(controller)
        attention = merge([controller_repeated, seq], mode='concat', concat_axis=-1)
        attention = TimeDistributed(Dense(1, activation='sigmoid'))(attention)
        attention = Flatten()(attention)
        attention = Lambda(to_prob, output_shape=(self._config['context_window_size'],))(attention)

        attention_repeated = RepeatVector(self._w2v.conceptEmbeddingsSz)(attention)
        attention_repeated = Permute((2, 1))(attention_repeated)

        weighted = merge([attention_repeated, seq], mode='mul')
        summed = Lambda(sum_seq, output_shape=(self._w2v.conceptEmbeddingsSz,))(weighted)
        return summed

    def addContextInput(self, controller1=None, controller2=None):
        left_context_input = Input(shape=(self._config['context_window_size'],), dtype='int32', name='left_context_input')
        right_context_input = Input(shape=(self._config['context_window_size'],), dtype='int32', name='right_context_input')
        self.inputs += [left_context_input, right_context_input]
        left_context_embed = self.word_embed_layer(left_context_input)
        right_context_embed = self.word_embed_layer(right_context_input)


        if self._config['context_network'] == 'gru':
            left_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu')(left_context_embed)
            right_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu')(right_context_embed)
            self.to_join += [left_rnn, right_rnn]
        elif self._config['context_network'] == 'mean':
            left_mean = Lambda(nonzero_mean, output_shape=(self._w2v.conceptEmbeddingsSz,))(left_context_embed)
            right_mean = Lambda(nonzero_mean, output_shape=(self._w2v.conceptEmbeddingsSz,))(right_context_embed)
            self.to_join += [left_mean, right_mean]
        elif self._config['context_network'] == 'attention':
            left_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu', return_sequences=True)(left_context_embed)
            right_rnn = GRU(self._w2v.wordEmbeddingsSz, activation='relu', return_sequences=True)(right_context_embed)

            attention_left1 = self.buildAttention(left_rnn, controller1)
            attention_right1 = self.buildAttention(right_rnn, controller1)
            self.to_join += [attention_left1, attention_right1]

            if controller2 is not None:
                attention_left2 = self.buildAttention(left_rnn, controller2)
                attention_right2 = self.buildAttention(right_rnn, controller2)
                self.to_join += [attention_left2, attention_right2]
        else:
            raise "unknown"

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
        self.train_loss = []
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
        model_builder = ModelBuilder(self._config, self._w2v)

        # use candidates input if they were specifically specified, or if we are using an attention network to process
        # the context.
        if 'candidates' in self.inputs or \
                ('context' in self.inputs and self._config['context_network'] == 'attention'):
            candidate1 = model_builder.addCandidateInput('candidate1_input', to_join='candidates' in self.inputs)
            if self._config['pairwise']:
                candidate2 = model_builder.addCandidateInput('candidate2_input', to_join='candidates' in self.inputs)
            else:
                candidate2 = None

        if 'context' in self.inputs:
            model_builder.addContextInput(controller1=candidate1, controller2=candidate2)

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

    def _2vec(self, mention, candidate1, candidate2):
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
            candidate1_X = np.array([self._w2v.conceptDict[candidate1]]) if candidate1 is not None else None
            candidate2_X = np.array([self._w2v.conceptDict[candidate2]]) if candidate2 is not None else None

        # get context input
        if 'context' in self.inputs:
            left_context_X = self.wordIteratorToIndices(mention.left_context_iter(),
                                                        self._config['context_window_size'])
            right_context_X = self.wordIteratorToIndices(mention.right_context_iter(),
                                                         self._config['context_window_size'])

        # get mention input
        if 'mention' in self.inputs:
            mention_X = self.wordIteratorToIndices(mention.mention_text_tokenized(),
                                                   self._config['max_mention_words'])

        if 'extra_features' in self.inputs:
            if self._config['pairwise']:
                extra_features_X = \
                    np.array(self._feature_generator.getPairwiseFeatures(mention, candidate1, candidate2))
            else:
                extra_features_X = \
                    np.array(self._feature_generator.getPointwiseFeatures(mention, candidate1))

        return left_context_X, right_context_X, mention_X, candidate1_X, candidate2_X, extra_features_X

    def wordIteratorToIndices(self, it, output_len):
        o = []
        for i, w in enumerate(it):
            if i >= output_len:
                break
            if w in self._w2v.wordDict and (self._stopwords is None or w not in self._stopwords):
                o.append(self._w2v.wordDict[w])
        if len(o) == 0:
            o.append(self._w2v.wordDict[self._w2v.DUMMY_KEY])
        o = o[:: -1]
        arr = np.zeros((output_len,))
        n = len(o) if len(o) <= output_len else output_len
        arr[:n] = np.array(o)[:n]
        return arr

    def train(self, mention, candidate1, candidate2, correct):
        """
        Takes a single example to train
        :param mention:    The mention to train on
        :param candidate1:  the first candidate
        :param candidate2:  the second candidate
        :param correct:     which of the two is correct (expected output)
        """
        vecs = self._2vec(mention, candidate1, candidate2)
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

            loss = self.model.train_on_batch(batchX, batchY)
            self.train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_mention_X = []
            self._batch_candidate1_X = []
            self._batch_candidate2_X = []
            self._batch_extra_features_X = []
            self._batchY = []

    def plotTrainLoss(self,fname, st=0):
        plt.plot(self.train_loss[st:])
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
            X['left_context_input'] = left_X.reshape((1, left_X.shape[0],))
            X['right_context_input'] = right_X.reshape((1, right_X.shape[0],))
        if 'mention' in self.inputs:
            X['mention_input'] = mention_X.reshape((1, mention_X.shape[0],))
        if 'extra_features' in self.inputs:
            X['extra_features_input'] = np.array(extraFeatures_X.reshape(1,extraFeatures_X.shape[0]))

        Y = self.model.predict(X, batch_size=1)
        return Y[0][0]
