from keras.layers import *
from FeatureGenerator import *
from PointwisePredict import *
from PairwisePredict import *
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class GBRTModel:
    def __init__(self, config, db=None, stats=None):
        if type(config) == str or type(config) == unicode:
            with open(config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config
        print "GBRT params:", self._config['hyper_patameters']
        self._model = GradientBoostingClassifier(loss=self._config['hyper_patameters']['loss'],
                                                 learning_rate=self._config['hyper_patameters']['learning_rate'],
                                                 n_estimators=self._config['hyper_patameters']['n_estimators'],
                                                 max_depth=self._config['hyper_patameters']['max_depth'],
                                                 max_features=None)

        self._feature_generator = FeatureGenerator(mention_features=self._config['features']['mention_features'],
                                                   entity_features=self._config['features']['entity_features'],
                                                   stats=stats,
                                                   db=db)
        self._train_X = []
        self._train_Y = []

    def getPredictor(self):
        return PointwisePredict(self)

    def predict(self, mention, candidate1, candidate2=None):
        if candidate2 is not None:
            raise "Unsupported operation"
        # create feature_vec from mention and candidate and predic prob for pointwise predictor
        feature_vec = self._feature_generator.getPointwiseFeatures(mention, candidate1)
        Y = self._model.predict_proba(np.asarray(feature_vec).reshape(1, -1))
        return Y[0][1]

    def train(self, mention, candidate1, candidate2, correct):
        '''
        Gathers mention and candidate features into a dataFrame
        :param mention:
        :param candidate1: suppose to be None
        :param candidate2: None
        :param correct:
        :return: only builds the _train_df
        '''
        self._train_X.append(self._feature_generator.getPointwiseFeatures(mention, candidate1))
        self._train_Y.append(1.0 if correct == candidate1 else -1.0)

    def finalize(self):
        '''
        trains the model over accumulated _train_df
        :return:
        '''

        trainX = np.array(self._train_X)
        trainy = np.array(self._train_Y)
        print "fitting gbrt model (", len(self._train_Y), "samples)"
        self._model.fit(trainX, trainy)

    def saveModel(self, fname):
        np.pickle.dump(self._model, open(fname, "wb"))

    def loadModel(self, fname):
        self._model = np.pickle.load(open(fname, "rb"))
