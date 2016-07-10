from WikilinksIterator import *
from WikilinksStatistics import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from Word2vecLoader import *

# Try and create an LSTM model

path = "C:\\repo\\DeepProject"
train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))

print 'Load embeddings...'
w2v = Word2vecLoader(wordsFilePath=path+"\\data\\word2vec\\dim300vecs",
                     conceptsFilePath=path+"\\data\\word2vec\\dim300context_vecs")
wD = train_stats.mentionLinks
cD = train_stats.conceptCounts
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'done'

model = Sequential()
model.add(LSTM(w2v.embeddingSize, return_sequences=False, input_shape=(10, w2v.embeddingSize)))
model.add(Dense(w2v.embeddingSize))
model.add(Activation('tanh'))
model.compile(loss='square',optimizer='adagrad')

# let's try to regress the w2v vector of the next word
for wikilink in iter_train.wikilinks():

