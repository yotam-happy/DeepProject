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
path = "C:\\repo\\DeepProject"
if(not os.path.isdir(path)):
    path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\small_train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\small_evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
print "Done!"

print 'Loading embeddings...'
w2v = Word2vecLoader(wordsFilePath=path+"\\data\\word2vec\\dim300vecs",
                     conceptsFilePath=path+"\\data\\word2vec\\dim300context_vecs")
wD = train_stats.mentionLinks
cD = train_stats.conceptCounts
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(w2v.wordEmbeddings)
print 'conceptEmbeddings dict size: ',len(w2v.conceptEmbeddings)
print 'Done!'

"""
Training double gru model
"""

## TRAIN DEBUGGING CELL
print 'Training...'
pairwise_model = RNNPairwiseModel(w2v)
knockout_model = KnockoutModel(pairwise_model,train_stats)
#pairwise_model.loadModel(path + "\\models\\rnn")

trainer = ModelTrainer(iter_train, train_stats, pairwise_model, epochs=1)
evaluation_loss = []
for train_session in xrange(40):
    # train
    print "Training... ", train_session
    trainer.train()

    # evaluate
    print "Evaluating...", train_session
    evaluation = Evaluation(iter_eval,knockout_model)
    evaluation.evaluate()
    evaluation_loss.append(evaluation.precision())

    # save
    print "Saving...", train_session
    precision_f = open(path + "\\models\\rnn.precision.txt","a")
    precision_f.write(str(train_session) + ": " + str(evaluation.precision()) + "\n")
    precision_f.close()

    pairwise_model.saveModel(path + "\\models\\rnn." + str(train_session))

## Plot results
pairwise_model.plotTrainLoss()

##
"""
Training single gru model
"""

print 'Training...'
pairwise_model = ModelSingleGRU(w2v = w2v,context_window_sz=20)
knockout_model = KnockoutModel(pairwise_model,train_stats)

trainer = ModelTrainer(iter_train, train_stats, pairwise_model, epochs=1)
evaluation_loss = []
for train_session in xrange(40):
    # train
    print "Training... ", train_session
    trainer.train()

    # evaluate
    print "Evaluating...", train_session
    evaluation = Evaluation(iter_eval,knockout_model)
    evaluation.evaluate()
    evaluation_loss.append(evaluation.precision())

    # save
    print "Saving...", train_session
    precision_f = open(path + "\\models\\rnn_gru.precision.txt","a")
    precision_f.write(str(train_session) + ": " + str(evaluation.precision()) + "\n")
    precision_f.close()

    pairwise_model.saveModel(path + "\\models\\rnn_gru." + str(train_session))

## Plot results
pairwise_model.plotTrainLoss()
