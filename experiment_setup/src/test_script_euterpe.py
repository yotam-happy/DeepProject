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
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from ModelSingleGRU import ModelSingleGRU
from models.VanilllaNNPairwiseModel import *
from models.RNNPairwiseModel import *
from models.RNNFineTunePairwiseModel import *
from KnockoutModel import *
from WikilinksIterator import *
from WikilinksStatistics import *
from Word2vecLoader import *
from Evaluation import *
from ModelTrainer import *
import pickle
from ModelRunner import *
##

_path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if not os.path.isdir(_path):
    _path = "/home/noambox/DeepProject"
elif(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

test_flag = 0

# train on wikipedia intra-links corpus
print 'training on intra-wiki!'
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intrawiki/train-stats")
_iter_train = WikilinksNewIterator(_path + "/data/intrawiki/train", mention_filter=_train_stats.getGoodMentionsToDisambiguate())
_iter_eval = None

# train on wikipedia intra-links corpus
# print 'training on intra!'
# _train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
# _iter_train = WikilinksNewIterator(_path + "/data/intralinks/train-filtered",mention_filter=_train_stats.getGoodMentionsToDisambiguate())
# _iter_eval = None

# print 'training on wikilinks small!'
# _train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
# _iter_train = WikilinksNewIterator(_path+"/data/wikilinks/filtered/train")
# _iter_eval = WikilinksNewIterator(_path+"/data/wikilinks/filtered/validation")
print "Done!"

print 'Loading embeddings 500 dim embeddings...'
_w2v = Word2vecLoader(wordsFilePath=_path+"/data/word2vec/dim500vecs",
                     conceptsFilePath=_path+"/data/word2vec/dim500context_vecs")
wD = _train_stats.contextDictionary
cD = _train_stats.conceptCounts
_w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
#_w2v.randomEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(_w2v.wordEmbeddings), " wanted: ", len(wD)
print 'conceptEmbeddings dict size: ',len(_w2v.conceptEmbeddings), " wanted", len(cD)
print 'Done!'


# DEFINING THE MODELS - please seperate each model by ## and
# save its parameters in the exel table

## MODEL VERSION_____________________________________________
# model_name = "rnn.relu"
# n_epoch = 20
# pairwise_model = RNNFineTuneEmbdPairwiseModel(_w2v)
## __________________________________________________________
# model_name = "rnn3_wide"
# n_epoch = 20
# pairwise_model = RNNFineTuneEmbdPairwiseModel(_w2v,dropout=1,context_window_sz=15,numof_neurons=[1e3, 100])
## __________________________________________________________
# patience = 3 # number of epochs to include in the early stopping criterion
# model_name = "rnn4_wide"
# n_epoch = 20
# pairwise_model = RNNFineTuneEmbdPairwiseModel(_w2v,dropout=1,context_window_sz=15)
## ________________________________________________________
# _model_name = "small"
# _patience = 4 # number of epochs to include in the early stopping criterion
# _pairwise_model = RNNPairwiseModel(_w2v, _train_stats, addPriorFeature=True, dropout=0.2)
# _n_epoch = 20
## ________________________________________________________
# _patience = 5 # number of epochs to include in the early stopping criterion
# # _pairwise_model = RNNPairwiseModel(_w2v, _train_stats, addPriorFeature=True, dropout=0.2)
# _pairwise_model = RNNPairwiseModel(_w2v, _train_stats, dropout=0.2)
# _n_epoch = 10
# _model_name = 'intra_patience_' + _patience.__str__() + '_n_epoch_'+ _n_epoch.__str__()
# model_runner = ModelRunner(_pairwise_model , _model_name , _iter_train, _iter_eval, _train_stats, _path ,n_epoch = _n_epoch, patience= _patience, filterWords = True, p= 1)
## ________________________________________________________
_patience = None
_pairwise_model = RNNPairwiseModel(_w2v, _train_stats, dropout=0.2)
_n_epoch = 20
_model_name = 'intrawiki_patience_' + _patience.__str__() + '_n_epoch_'+ _n_epoch.__str__() +'_500dimembeddings'
model_runner = ModelRunner(_pairwise_model , _model_name , _iter_train, _iter_eval, _train_stats, _path ,n_epoch = _n_epoch, patience= _patience, filterWords = True, p= 1)
## ________________________________________________________
# _model_name = "tiny_intra"
# _patience = 1 # number of epochs to include in the early stopping criterion
# _pairwise_model = RNNPairwiseModel(_w2v, _train_stats, addPriorFeature=True, dropout=0.2)
# _n_epoch = 1
# _p = 0.01
# model_runner = ModelRunner(_pairwise_model , _model_name , _iter_train, _iter_eval, _train_stats, _path ,n_epoch = _n_epoch, patience= _patience, filterWords = True, p= _p)

## TRAIN DEBUGGING CELL
# model_runner = ModelRunner(_pairwise_model , _model_name , _iter_train, _iter_eval, _train_stats, _path ,n_epoch = _n_epoch, patience= _patience, filterWords = True)
model_runner.run(doEvaluation = False)
print "Finished TRAINING!!"

## TESTING
if test_flag:
    print "Evaluating after training!"
    iter_test = WikilinksNewIterator(_path + "/data/intrawiki/test",
                                       mention_filter=_train_stats.getGoodMentionsToDisambiguate())
    knockout_model = KnockoutModel(model_runner.pairwise_model, _train_stats)
    evaluation = Evaluation(iter_test, knockout_model)
    evaluation.evaluate()