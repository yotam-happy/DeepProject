"""
This class simplifies the whole training, evaluation and saving procedure
It includes parameters for running a model. replaces the traditional fit and structure of
keras
"""
import numpy as np

from Evaluation import Evaluation
from KnockoutModel import KnockoutModel
from ModelTrainer import ModelTrainer


class ModelRunner:
        def __init__(self, model , model_name , itr_train, itr_eval, stats, path,n_epoch = 20, patience = None,  filterWords = False, filterSenses = False, p=1.0):
            '''
            :param model_name: all saved files start with this name
            :param path:            path to save files
            :param model:  pairwise model to train
            :param train_stats:     stats object of the training set
            :param itr_train:      iterator for the training set
            :param itr_eval:       iterator for the evaluation set
            :param filterWords:     True if we wish to treat 10% of the words as unseen (filter them from the training set)
            :param filterSenses:    True if we wish to treat 10% of the words, and all possible senses for them as unseen
                                    (filter them from the training set)
            :param doEvaluation:    False if we want only training, without evaluation after every train session
            :param p:               fraction of words to train/test on (reduce the problem size)
            '''
            self.stats = stats
            self.itr_train = itr_train
            self.itr_eval = itr_eval
            self.pairwise_model = model
            self.model_name = model_name
            self.n_epoch = n_epoch
            self.patience = patience
            self.path = path
            self.filterSenses = filterSenses
            self.filterWords = filterWords
            self.p = p
            self.evaluation_loss = []

        def run(self, doEvaluation = True):
            knockout_model = KnockoutModel(self.pairwise_model,self.stats)
            #pairwise_model.loadModel(path + "\\models\\rnn")

            wordsForBroblem = self.stats.getRandomWordSubset(self.p)
            wordFilter = self.stats.getRandomWordSubset(0.1,baseSubset=wordsForBroblem) if self.filterWords or self.filterSenses else None
            senseFilter = self.stats.getSensesFor(wordFilter) if self.filterSenses else None

            knockout_model = KnockoutModel(self.pairwise_model, self.stats)
            trainer = ModelTrainer(self.itr_train, self.stats, self.pairwise_model, epochs=1, wordInclude=wordsForBroblem,wordExclude=wordFilter, senseFilter=senseFilter)

            for train_session in xrange(self.n_epoch):
                # train
                print "Training... ", train_session
                trainer.train()

                self.pairwise_model.saveModel(self.path + "/models/" + self.model_name + "." + str(train_session) + ".out")

                if doEvaluation:
                    self.evaluate(train_session, knockout_model, self.evaluation_loss ,wordInclude=wordsForBroblem, wordExclude=wordFilter, name = self.model_name + ".eval")
                    if self.filterWords or self.filterSenses:
                        self.evaluate(train_session, knockout_model, self.evaluation_loss, wordInclude=wordFilter,name = self.model_name + ".unseen.eval")

                # early stopping criterion
                if self.patience is not None and train_session > self.patience and doEvaluation == 1 and \
                        np.all([val_loss <= self.evaluation_loss[-1:] for val_loss in self.evaluation_loss[( -1 - self.patience ):]]):
                    print 'patience criteria activated! exits training... '
                    break

            ## Plot train loss to file
            print "done training!"
            self.pairwise_model.plotTrainLoss(self.path + "/models/" + self.model_name + ".train_loss.png", st=10)
            self.evaluation_loss = []

        def evaluate(self, t_session, knockout_model, evaluation_loss, wordExclude=None, wordInclude=None, name = None):
            if name is None:
                name = self.model_name

            # evaluate
            print "Evaluating...", t_session
            evaluation = Evaluation(self.itr_eval, knockout_model, wordExcludeFilter=wordExclude,wordIncludeFilter=wordInclude, stats= self.stats)
            evaluation.evaluate()
            evaluation_loss.append(evaluation.precision())

            # save evalutation
            print "Saving...", t_session
            precision_f = open(self.path + "/models/"+name+".precision.txt","a")
            precision_f.write(str(t_session) + "train: " + str(evaluation.precision()) + "\n")
            precision_f.close()



# def eval(experiment_name, path, train_session_nr, knockout_model, iter_eval, wordExclude=None, wordInclude=None, stats=None):
#     # evaluate
#     print "Evaluating " + experiment_name + "...", train_session_nr
#     evaluation = Evaluation(iter_eval, knockout_model, wordExcludeFilter=wordExclude, wordIncludeFilter=wordInclude, stats=stats)
#     evaluation.evaluate()
#
#     # save
#     print "Saving...", train_session_nr
#     precision_f = open(path + "/models/" + experiment_name + ".precision.txt", "a")
#     precision_f.write(str(train_session_nr) + " train: " + str(evaluation.precision()) + "\n")
#     precision_f.close()
#
#
# def experiment(experiment_name, path, pairwise_model, train_stats, iter_train, iter_eval, filterWords = False, filterSenses = False, doEvaluation=True, p=1.0):
#     '''

#
#     wordsForBroblem = train_stats.getRandomWordSubset(p)
#     wordFilter = train_stats.getRandomWordSubset(0.1, baseSubset=wordsForBroblem) if filterWords or filterSenses else None
#     senseFilter = train_stats.getSensesFor(wordFilter) if filterSenses else None
#
#     knockout_model = KnockoutModel(pairwise_model, train_stats)
#     trainer = ModelTrainer(iter_train, train_stats, pairwise_model, epochs=1, wordInclude=wordsForBroblem, wordExclude=wordFilter, senseFilter=senseFilter)
#     for train_session in xrange(20):
#         # train
#         print "Training... ", train_session
#         trainer.train()
#
#         pairwise_model.saveModel(path + "/models/" + experiment_name + "." + str(train_session) +  ".out")
#
#         if doEvaluation:
#             eval(experiment_name + ".eval", path, train_session, knockout_model, iter_eval, wordInclude=wordsForBroblem, wordExclude=wordFilter, stats=train_stats)
#             if filterWords or filterSenses:
#                 eval(experiment_name + ".unseen.eval", path, train_session, knockout_model, iter_eval, wordInclude=wordFilter,stats=train_stats)
#
#
#     ## Plot train loss to file
#     pairwise_model.plotTrainLoss(path + "/models/" + experiment_name + ".train_loss.png", st=10)