import os
import re
import numpy
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting
import datetime as dt # module for manipulating dates and times
import numpy.linalg as lin # module for performing linear algebra operationsb
from zipfile import ZipFile
import json

import WikilinksIterator
import Word2vecLoader
from WikilinksIterator import WikilinksOldIterator

class DesaModel:
# the depp_ESA model file. Enables model initialization, training and prediction

    def __init__(self, Word2vecLoader = "wvl", WikilinksIterator = "witr", SenseDic = "sdic"):
        # initialization of database path
        self.wvl = Word2vecLoader
        self.witr = WikilinksOldIterator
        self.sense_dic = SenseDic

    def train(self):
    # iterates using the wikilinkIterator over all possible examples when with data
    # structure - wlink, S (all senses)
        stats = WikilinksIterator.WikilinksStatistics(self.witr)
        stats.senseDicCreation()
        for wlink in self.witr.get_wlinks():
            print wlink['word'],'\n', wlink['right_context'],'\n',wlink['left_context'],'\n',wlink['wikiId'],'\n',stats.senseDic[wlink['word']]
            self.train_for_sample(wlink,stats.senseDic[wlink['word']])

    def train_for_sample(self, wlink, S):
        # TODO: ommits right sense from set of senses and train for each wrong sense
        # versus the wright sense for knockout modle
        return

    # def predict(self):
    #     # TODO: recieves data set and prodicts accuracy according to model

if __name__ == '__main__':
    path = os.path.split(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))[0]
    witr = WikilinksIterator.WikilinksOldIterator(path+'\Data',  limit_files = 1)
    stats = WikilinksIterator.WikilinksStatistics(witr)
    stats.senseDicCreation()

    # stats.calcStatistics()
    # dsm = DesaModel(WikilinksIterator= witr)
    for wlink in witr.get_wlinks():
        print wlink['word'],'\n', wlink['right_context'],'\n',wlink['left_context'],'\n',wlink['wikiId'],'\n',stats.senseDic[wlink['word']]


