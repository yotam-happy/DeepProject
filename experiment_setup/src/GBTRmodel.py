from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor


class GBTRmodel:

    def __init__(self, itr = None, stats = None, db = None, feature_generator = None, n_epoch = None, model = None):
        self._stats = stats
        self._iter =itr
        self._db = db
        self._n_epoch = n_epoch
        if model:
            self._clf = model
        else:
            # initialize GBTR (it is the classification model and not the regresion because the
            # logistic regression loss)
            self._clf = GradientBoostingRegressor(loss = 'huber', learning_rate= 0.02, n_estimators=10000, max_depth=3, max_features=None) # TODO: Check deviance loss with GBTC

    def fit(self,  data, label):
        # train individually on every mention appearing in that doc using the _model
        self._clf.fit(X = data, y = label)
        return

    def predict(self,data):
        self._clf.predict(data)

    def evaluate(self, data, label):
        metrics.accuracy_score(y_pred= self.predict(data), y_true=label)

    def train_batch(self):
        # TOD
        return