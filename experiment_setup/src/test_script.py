"""
Mannually execute each class dependency to work with the project without compiling
Best and easiest way to debug code in python. You can actully view to whole workspace

ATTENTION: Change the path variable in the end of the file
ATTENTION: If you de modificaitons in the classes please keep the copyies here updated! (There must be a better way but I am lazy)

NOTEs: use collapse all shortcut CTRL + SHIFT + NumPad (-)  to navigate and excecute easily the code.
also use the ALT+SHIFT+E to execute single lines or whole code fragments
I also recommend on Pycharm cell mode plugin for easier execution of code fragments
(Noam)
"""

## The cell seperator
import os

from ModelSingleGRU import ModelSingleGRU
from VanilllaNNPairwiseModel import *
from RNNModel import *
from RNNModelFineTuneEmbd import *
from KnockoutModel import *
from WikilinksIterator import *
from WikilinksStatistics import *
from Word2vecLoader import *
from Evaluation import *
from ModelTrainer import *
import pickle
import ModelRunner
import nltk
##
"""
here we test the VanillaNN structure
This is the main script
"""

path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if(not os.path.isdir(path)):
    path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

train_stats = WikilinksStatistics(None, load_from_file_path=path+"/data/wikilinks/small/wikilinks.stats")
iter_train = WikilinksNewIterator(path+"/data/wikilinks/small_train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
iter_eval = WikilinksNewIterator(path+"/data/wikilinks/small_evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
print "Done!"

print 'Loading embeddings...'
w2v = Word2vecLoader(wordsFilePath=path+"/data/word2vec/dim300vecs",
                     conceptsFilePath=path+"/data/word2vec/dim300context_vecs")
wD = train_stats.contextDictionary
cD = train_stats.conceptCounts
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(w2v.wordEmbeddings)
print 'conceptEmbeddings dict size: ',len(w2v.conceptEmbeddings)
print 'Done!'

"""
DEFINING THE MODELS - please seperate each model by ## and
save its parameters in the excel table
"""

## MODEL VERSION_____________________________________________
model_name = "rnn.relu"
n_epoch = 20
pairwise_model = RNNFineTuneEmbdPairwiseModel(w2v)

## __________________________________________________________
model_name = "rnn3_wide"
n_epoch = 20
pairwise_model = RNNFineTuneEmbdPairwiseModel(w2v,dropout=1,context_window_sz=15,numof_neurons=[1e3, 100])
## __________________________________________________________
patience = 3 # number of epochs to include in the early stopping criterion
model_name = "rnn4_wordFilter"
n_epoch = 20
pairwise_model = RNNFineTuneEmbdPairwiseModel(w2v,dropout=1,context_window_sz=15)

## TRAIN DEBUGGING CELL
model_runner = ModelRunner(model = pairwise_model, model_name = model_name, n_epoch = n_epoch, iterator = iter_train, stats = train_stats, path = path, patience= None)
model_runner.run()

## Plot results

pairwise_model.plotTrainLoss()

## Save model

pairwise_model.saveModel(model_name)