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
save its parameters in the exel table
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

## TRAIN DEBUGGING CELL
model_runner = ModelRunner(model = pairwise_model, model_name = model_name, n_epoch = n_epoch, iterator = iter_train, stats = train_stats, path = path, patience= None)
ModelRunner.run()

##
# print 'Training...'
#
# #pairwise_model = RNNFineTuneEmbdPairwiseModel(w2v)
# #pairwise_model = RNNPairwiseModel(w2v)
# #pairwise_model = ModelSingleGRU(w2v)
# #pairwise_model = VanillaNNPairwiseModel(w2v)
#
# knockout_model = KnockoutModel(pairwise_model,train_stats)
# #pairwise_model.loadModel(path + "\\models\\rnn")
#
# trainer = ModelTrainer(iter_train, train_stats, pairwise_model, epochs=1)
#
# evaluation_loss = []
# for train_session in xrange(n_epoch):
#     # train
#     print "Training... ", train_session
#     trainer.train()
#
#     # evaluate
#     print "Evaluating...", train_session
#     evaluation = Evaluation(iter_eval,knockout_model)
#     evaluation.evaluate()
#     evaluation_loss.append(evaluation.precision())
#
#     # save evalutation
#     print "Saving...", train_session
#     precision_f = open(path + "\\models\\"+model_name+".precision.txt","a")
#     precision_f.write(str(train_session) + ": " + str(evaluation.precision()) + "\n")
#     precision_f.close()
#
#     train_loss_f = open(path + "\\models\\"+model_name+".train_loss.txt",'wb')
#     pickle.dump(pairwise_model._train_loss, train_loss_f)
#     train_loss_f.close()
#
#     # early stopping criterion
#     if train_session > patience and np.all([val_loss <= evaluation_loss[-1:] for val_loss in evaluation_loss[-1-patience:]]):
#         break

## Plot results

pairwise_model.plotTrainLoss()

## Save model

pairwise_model.saveModel(model_name)