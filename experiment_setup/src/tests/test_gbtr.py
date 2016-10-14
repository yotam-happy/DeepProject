from GBTRmodel import GBTRmodel
import pandas as pd
from ProjectSettings import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

path, pc_name = getPath()
test_set = 'testb'
train = pd.read_pickle(path + "/data/CoNLL/train_data_base_conll")
test = pd.read_pickle(path + "/data/CoNLL/"+test_set+"_data_base_conll")

feature_names = filter(lambda name: name != 'label',train.columns)
trainX = train[feature_names]
trainy = (train['label'] + 1)/2

testX = test[feature_names]
testy = (test['label'] + 1)/2

# training
print 'training...'
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.02, n_estimators=10000,
                                max_depth=3,max_features=None)
model.fit(trainX.as_matrix(),trainy.as_matrix())
# gbtr = GBTRmodel(model=clf) # I don't know why but using a different GBTR class makes problems
# gbtr.fit(trainX.as_matrix(),trainy.as_matrix())
print 'Done training!'

# predict
# print model.predict(trainX.as_matrix()[:20,:])
# print 'train accuracy: ',gbtr.evaluate(trainX.as_matrix() > 0.5,trainy.as_matrix())
# print 'testa accuracy: ',gbtr.evaluate(testX.as_matrix() > 0.5, testy.as_matrix())

# evaluate
pred = model.predict_proba(testX.as_matrix())
print test_set+' accuracy: ',\
    metrics.accuracy_score(y_pred=pred[:,1] > 0.5, y_true=testy.as_matrix())

## validate data size
print 'number of non-NIL mentions in train: ',len(filter(lambda x: x==1,trainy.as_matrix())),' should be : 18505 according to Chisholm et al.'
print 'number of non-NIL mentions in test: ',len(filter(lambda x: x==1,testy.as_matrix())),' should be : 4485 according to Chisholm et al.'

## feature importance
importances = model.feature_importances_
std = np.std([tree[0].feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(trainX.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances for yamada's baseline features")
plt.bar(range(trainX.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(trainX.shape[1]), indices)
plt.xlim([-1, trainX.shape[1]])
ax = plt.axes()
ax.set_xticklabels([feature_names[i] for i in indices])
plt.show()

## ROC curve
fpr, tpr, thresholds = metrics.roc_curve(testy.as_matrix(), pred[:,1] , pos_label=1 )
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

