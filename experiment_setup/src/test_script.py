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

from Evaluation import *
from KnockoutModel import *
from ModelTrainer import *
from models.RNNPairwiseModel import *
from models.BaselinePairwiseModel import *

##

def eval(experiment_name, path, train_session_nr, knockout_model, iter_eval, wordExclude=None, wordInclude=None, stats=None, sampling=None):
    # evaluate
    print "Evaluating " + experiment_name + "...", train_session_nr
    evaluation = Evaluation(iter_eval, knockout_model, wordExcludeFilter=wordExclude, wordIncludeFilter=wordInclude, stats=stats, sampling=sampling)
    evaluation.evaluate()

    # save
    print "Saving...", train_session_nr
    precision_f = open(path + "/models/" + experiment_name + ".precision.txt", "a")
    precision_f.write(str(train_session_nr) + " train: " + str(evaluation.precision()) + "\n")
    precision_f.close()


def experiment(experiment_name, path, pairwise_model, train_stats, iter_train, iter_eval, filterWords = False, filterSenses = False, doEvaluation=True, p=1.0):
    '''
    :param experiment_name: all saved files start with this name
    :param path:            path to save files
    :param pairwise_model:  pairwise model to train
    :param train_stats:     stats object of the training set
    :param iter_train:      iterator for the training set
    :param iter_eval:       iterator for the evaluation set
    :param filterWords:     True if we wish to treat 10% of the words as unseen (filter them from the training set)
    :param filterSenses:    True if we wish to treat 10% of the words, and all possible senses for them as unseen
                            (filter them from the training set)
    :param doEvaluation:    False if we want only training, without evaluation after every train session
    :param p:               fraction of words to train/test on (reduce the problem size)
    '''

    wordsForBroblem = train_stats.getRandomWordSubset(p)
    wordFilter = train_stats.getRandomWordSubset(0.1, baseSubset=wordsForBroblem) if filterWords or filterSenses else None
    senseFilter = train_stats.getSensesFor(wordFilter) if filterSenses else None

    knockout_model = KnockoutModel(pairwise_model, train_stats)
    trainer = ModelTrainer(iter_train, train_stats, pairwise_model, epochs=1, wordInclude=wordsForBroblem, wordExclude=wordFilter, senseFilter=senseFilter)
    for train_session in xrange(200):
        # train
        print "Training... ", train_session
        trainer.train()

        pairwise_model.saveModel(path + "/models/" + experiment_name + "." + str(train_session) +  ".out")

        if doEvaluation:
            eval(experiment_name + ".eval", path, train_session, knockout_model, iter_eval, wordInclude=wordsForBroblem, wordExclude=wordFilter, stats=train_stats, sampling=0.005)
            if filterWords or filterSenses:
                eval(experiment_name + ".unseen.eval", path, train_session, knockout_model, iter_eval, wordInclude=wordFilter,stats=train_stats, sampling=0.05)


    ## Plot train loss to file
    pairwise_model.plotTrainLoss(path + "/models/" + experiment_name + ".train_loss.png", st=10)

##

_path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_iter_train = WikilinksNewIterator(_path+"/data/intralinks/train-filtered")
#_iter_eval = WikilinksNewIterator(_path+"/data/intralinks/test-filtered")

_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
_iter_train = WikilinksNewIterator(_path+"/data/wikilinks/fixed/train")
_iter_eval = WikilinksNewIterator(_path+"/data/wikilinks/fixed/evaluation")
print "Done!"

print 'Loading embeddings...'
_w2v = Word2vecLoader(wordsFilePath=_path+"/data/word2vec/dim300vecs",
                     conceptsFilePath=_path+"/data/word2vec/dim300context_vecs")
wD = _train_stats.contextDictionary
cD = _train_stats.conceptCounts
_w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
#_w2v.randomEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(_w2v.wordEmbeddings), " wanted: ", len(wD)
print 'conceptEmbeddings dict size: ',len(_w2v.conceptEmbeddings), " wanted", len(cD)
print 'Done!'

"""
Training double gru model
"""

## TRAIN DEBUGGING CELL
print 'Training...'

_feature_generator = FeatureGenerator(entity_features={'log_prior', 'cond_prior'}, stats=_train_stats)
#_pairwise_model = RNNFineTuneEmbdPairwiseModel(_w2v, dropout=0.1)
_pairwise_model = RNNFineTunePairwiseModel(_w2v, dropout=0.1, feature_generator=_feature_generator)
#_pairwise_model = VanillaNNPairwiseModel(_w2v)
#_pairwise_model.loadModel(_path + "/models/model.10.out")

experiment("small", _path, _pairwise_model, _train_stats, _iter_train, _iter_eval, doEvaluation=False, filterWords=True)

## baseline
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
#_iter_test = WikilinksNewIterator(_path+"/data/wikilinks/fixed/evaluation")
#_pairwise_model = BaselinePairwiseModel(_train_stats)
#_pairwise_model = GuessPairwiseModel()
#knockout_model = KnockoutModel(_pairwise_model, _train_stats)
#evaluation = Evaluation(_iter_test, knockout_model, stats=_train_stats)
#evaluation.evaluate()

