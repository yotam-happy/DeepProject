"""
This class simplifies the whole training, evaluation and saving procedure
It includes parameters for running a model. replaces the traditional fit and structure of
keras
"""
from Evaluation import Evaluation
from KnockoutModel import KnockoutModel
from ModelTrainer import ModelTrainer
import numpy as np
import pickle

class ModelRunner():
        def __init__(self, model, model_name, iterator, stats, path,n_epoch = 20, patience= None):
            self.stats = stats
            self.itr = iterator
            self.pairwise_model = model
            self.model_name = model_name
            self.n_epoch = n_epoch
            self.patiance = patience
            self.path = path

        def run(self):
            knockout_model = KnockoutModel(self.pairwise_model,self.train_stats)
            #pairwise_model.loadModel(path + "\\models\\rnn")

            trainer = ModelTrainer(self.itr, self.stats, self.pairwise_model, epochs=1)

            evaluation_loss = []
            for train_session in xrange(self.n_epoch):
                # train
                print "Training... ", train_session
                trainer.train()

                # evaluate
                print "Evaluating...", train_session
                evaluation = Evaluation(self.itr,knockout_model)
                evaluation.evaluate()
                evaluation_loss.append(evaluation.precision())

                # save evalutation
                print "Saving...", train_session
                precision_f = open(self.path + "\\models\\"+self.model_name+".precision.txt","a")
                precision_f.write(str(train_session) + ": " + str(evaluation.precision()) + "\n")
                precision_f.close()

                train_loss_f = open(self.path + "\\models\\"+self.model_name+".train_loss.txt",'wb')
                pickle.dump(self.pairwise_model._train_loss, train_loss_f)
                train_loss_f.close()

                # early stopping criterion
                if train_session > self.patience and np.all([val_loss <= evaluation_loss[-1:] for val_loss in evaluation_loss[-1-patience:]]):
                    break
