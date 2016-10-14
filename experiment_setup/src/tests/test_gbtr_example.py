from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

'''
This code comes to check if we can fit models with sklearn using fit()
instead of partial_fit() that does not exists in GBTR models. Conclusions
at the end
'''

class DataIterator:
    def __init__(self, data):
        # self._label = label
        self._data = data
        self.current = 0
        self.high = data.shape[0] - 1

    def __iter__(self):
        return self

    def next(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return np.reshape(self._data[self.current - 1][:],(1,-1))
                   # np.reshape(self._label[self.current - 1],(1,-1))

def DataIter(data, label_mode = False, batch_size = 100):
    current = 1
    high = data.shape[0] - 1
    while current <= high:
        if label_mode:
            yield np.ravel(data[current - 1:current+batch_size - 1][:])
        else:
            yield data[current - 1:current+batch_size - 1][:]
        current += batch_size+1


X, y = make_hastie_10_2(n_samples=10000)

## make model and fit
est_all = GradientBoostingClassifier(n_estimators=200, max_depth=3)
est_all.fit(X, y)

## test performance non-batch
# pred = est_all.predict(X)
regular_training = est_all.predict_proba(X)[:5]
print regular_training

## make model and use iterative fit
est_batch = GradientBoostingClassifier(n_estimators=200, max_depth=3)
x_iter =  DataIter(X)
y_iter = DataIter(y, label_mode=True)
for i in xrange(len(X)/100-1):
    # print (x_batch,y_batch)
    # y_iter.next()
    # print i
    est_batch.fit(x_iter.next(), np.ravel(y_iter.next()))
print 'done training'

## test performance batch
# pred = est_batch.predict(X)
batch_training = est_batch.predict_proba(X)[:5]
print batch_training

# conclusions - must have partial fit or somewhat get rid of the .next()?