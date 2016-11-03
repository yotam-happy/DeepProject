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
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class GBRTModel:
    def __init__(self, config, load_path = None, w2v = None, db = None, stats = None):

        if type(config) == str:
            with open(config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config

        self._model = GradientBoostingClassifier(loss= self._config['hyper_patameters']['loss'],
                                                 learning_rate=self._config['hyper_patameters']['learning_rate'],
                                                 n_estimators= self._config['hyper_patameters']['n_estimators'],
                                                 max_depth =  self._config['hyper_patameters']['max_depth'],
                                                 max_features= None)
        if 'feature_generator' in self._config:
            self._feature_generator = FeatureGenerator(mention_features=
                                                       self._config['feature_generator']['mention_features'],
                                                       entity_features=
                                                       self._config['feature_generator']['entity_features'],
                                                       stats=stats, db=db)
        self._train_df = pd.DataFrame()
        self.group_indx = 0.0

    def getPredictor(self):
        return PointwisePredict(self)

    def predict(self, mention, candidate):
        # create feature_vec from mention and candidate and predic prob for pointwise predictor
        feature_vec = self._feature_gen.getPointwiseFeatures(self, mention, candidate)
        Y = self._model.predict_prob( np.asarray(feature_vec))
        return Y

    def train(self, mention, candidate1, candidate2, correct):
        '''
        Gathers mention and candidate features into a dataFrame
        :param mention:
        :param candidate1: suppose to be None
        :param candidate2: None
        :param correct:
        :return: only builds the _train_df
        '''

        if self.group_indx == 0.0 and self._train_df.size() != 0.0:
            print 'GBRTModelError:: Training DataFrame for model was already build...'
            raise

        # update group index of current mention chunck ( changes only at the first sense input which is the golden sense )
        if mention is correct:
            self.group_indx += 1.0

        feature_vec = self._feature_gen.getPointwiseFeatures(self, mention, candidate1)
        feature_df = pd.DataFrame(feature_vec , index=candidate1, columns=self._feature_gen.getFeatureNames())
        feature_df['label'] = 1.0 if correct == candidate1 else 0.0
        feature_vec['group_indx'] = self.group_indx
        self._train_df = pd.concat([self._train_df, feature_vec])

    def finalize(self):
        '''
        trains the model over accumulated _train_df
        :return:
        '''

        feature_names = filter(lambda name: name not in ['label', 'group_indx'], self._train_df.columns)
        trainX = self._train_df[feature_names]
        trainy = (self._train_df['label'] + 1) / 2
        self._model.fit(trainX.as_matrix(), trainy.as_matrix()) # TODO: possible to train more than 1 epoch?
        self.group_indx = 0.0

    def saveModel(self, fname):
        np.pickle.dump(self._model, open(fname, "wb"))

    def loadModel(self, fname):
        self.model = np.pickle.load(open(fname,"rb"))
