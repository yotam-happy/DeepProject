from WikilinksIterator import *
from WikilinksStatistics import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from KnockoutModel import *
from Word2vecLoader import *
from Evaluation import *
import matplotlib.pyplot as plt

# Try and create an LSTM model

path = "C:\\repo\\DeepProject"
if(not os.path.isdir(path)):
    path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\small_train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))

print 'Load embeddings...'
w2v = Word2vecLoader(wordsFilePath=path+"\\data\\word2vec\\dim300vecs",
                     conceptsFilePath=path+"\\data\\word2vec\\dim300context_vecs")
wD = train_stats.mentionLinks
cD = train_stats.conceptCounts
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(w2v.wordEmbeddings)
print 'conceptEmbeddings dict size: ',len(w2v.conceptEmbeddings)
print 'done'

context_window = 10

model = Sequential()
model.add(LSTM(w2v.embeddingSize, return_sequences=False, input_shape=(context_window, w2v.embeddingSize)))
model.add(Dense(w2v.embeddingSize))
model.add(Activation('tanh'))
model.compile(loss='mse',optimizer='adagrad')

def wordListToVectors(w2v, l):
    o = []
    for w in l:
        if w in w2v.wordEmbeddings:
            o.append(w2v.wordEmbeddings[w])
    return np.asarray(o)

batch_X = []
batch_Y = []
train_loss = []
epochs = 10
# let's try to regress the w2v vector of the right sense
for epoch in xrange(epochs):
    for wikilink in iter_train.wikilinks():
        if 'left_context' in wikilink:
            ar = wordListToVectors(w2v, wikilink['left_context'])
            if ar.shape[0] < context_window:
                continue
            X = ar[-10:,:]
            Y = np.array(w2v.conceptEmbeddings[wikilink['wikiId']])
            batch_X.append(X)
            batch_Y.append(Y)
            if len(batch_X) >= 1024:
                batchX = np.array(batch_X)
                batchY = np.array(batch_Y)
                loss = model.train_on_batch(batchX, batchY)
                train_loss.append(loss)
                batch_X = []
                batch_Y = []
                print 'Done batch. Size of batch x - ', batchX.shape, '; loss: ', loss
    print "done epoch ", epoch



plt.plot(train_loss[100:])
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()

class mmooddeell:
    def __init__(self, model, w2v):
        self.model = model
        self._w2v = w2v

    def wordListToVectors(self, l):
        o = []
        for w in l:
            if w in self._w2v.wordEmbeddings:
                o.append(self._w2v.wordEmbeddings[w])
        return np.asarray(o)

    def _2vec(self, wikilink, candidate1, candidate2):
        """
        Transforms input to w2v vectors
        returns a tuple: (wikilink vec, candidate1 vec, candidate2 vec)

        if cannot produce wikilink vec or vectors for both candidates then returns None
        if cannot produce vector to only one of the candidates then returns the id of the other
        """
        if candidate1 not in self._w2v.conceptEmbeddings and candidate2 not in self._w2v.conceptEmbeddings:
            return None
        if 'left_context' not in wikilink:
            return None

        if candidate1 not in self._w2v.conceptEmbeddings:
            return candidate2
        if candidate2 not in self._w2v.conceptEmbeddings:
            return candidate1

        candidate1_vec = self._w2v.conceptEmbeddings[candidate1]
        candidate2_vec = self._w2v.conceptEmbeddings[candidate2]

        ar = wordListToVectors(w2v, wikilink['left_context'])
        if ar.shape[0] < context_window:
            return None
        return (ar[-10:,:], candidate1_vec, candidate2_vec)

    def predict(self, wikilink, candidate1, candidate2):
        vecs = self._2vec(wikilink, candidate1, candidate2)
        if not isinstance(vecs, tuple):
            return vecs
        (X, candidate1_vec, candidate2_vec) = vecs

        # cache the context embedding. We are likely to see the same wikilink a number of times
        if '_context_embed' not in wikilink:
            context_vec = self.model.predict(np.array([X]), batch_size=1)
            wikilink['_context_embed'] = context_vec
        else:
            context_vec = wikilink['_context_embed']

        if self._w2v.distance(context_vec, candidate1_vec) < self._w2v.distance(context_vec, candidate2_vec):
            return candidate1
        else:
            return candidate2

iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\small_evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
m = mmooddeell(model, w2v)
knockout_model = KnockoutModel(m,train_stats)
evaluation = Evaluation(iter_eval,knockout_model)
evaluation.evaluate()

## incorporate into the vanilla model (code copy pasted from test_script.py

## The cell seperator
import os
from VanilllaNNPairwiseModel import *
from KnockoutModel import *
from WikilinksIterator import *
from WikilinksStatistics import *
from Word2vecLoader import *
from Evaluation import *
from ModelTrainer import *
##

"""
here we test the VanillaNN structure
This is the main script
"""
print "Loading iterators+stats..."
os.chdir("C:\\repo\\DeepProject") # TODO: Yotam, you need to change this in order to work with this file
path = "C:\\repo\\DeepProject"
train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")

iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\small_train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\small_evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
print "Done!"

## TRAIN DEBUGGING CELL
print 'Training...'
pairwise_model = VanillaNNPairwiseModel(w2v, context_window_sz=10, lstm=model)
knockout_model = KnockoutModel(pairwise_model,train_stats)
#pairwise_model.loadModel(path + "\\models\\vanilla_nn_lstm")

trainer = ModelTrainer(iter_train, train_stats, pairwise_model, epochs=30)
trainer.train()
pairwise_model.saveModel(path + "\\models\\vanilla_nn_lstm")

## TEST
evaluation = Evaluation(iter_eval,knockout_model)
evaluation.evaluate()
pairwise_model.plotTrainLoss()
