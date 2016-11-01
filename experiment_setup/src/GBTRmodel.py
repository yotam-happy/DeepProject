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


class GBRTModel:
    def __init__(self, model, train_df = None, test_df = None):
        self._model = model
        self._train_df = train_df
        pass

    def getPredictor(self):
        return PointwisePredict(self)

    def predict(self, mention, candidate1, candidate2):
        # create feature_vec from mention and candidate
        Y = self._model.predict_prob(feature_vec)
        return Y

    def train(self, mention, candidate1, candidate2, correct):
        '''
        gathers mention and candidate features and trains the whole
        data structere with finalize in the end
        :param mention:
        :param candidate1:
        :param candidate2: None
        :param correct:
        :return:
        '''

        feature_vec
        pass

    def finalize(self):
        pass

    def saveModel(self, fname):
        pass

    def loadModel(self, fname):
        pass
