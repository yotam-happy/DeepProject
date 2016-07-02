import test
import os
import re
import numpy
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting
import datetime as dt # module for manipulating dates and times
import numpy.linalg as lin # module for performing linear algebra operationsb
from zipfile import ZipFile
import json

class desaModel:

    def __init__(self, wvl = "word2vecLoader "):
        # initialization of database path
        self.wvl = wvl

    # def sampleDataUnit(self, w = "word"):
    #     # collects all possible senses, S, and outputs it as (context, word, S, correct_s)
    #
    # def collectSenses(self, w = "words"):
    #     # extract a list off all senses according to input word
    #
    # def train(self):
    #     # currently empy
    #
    # def predict(self):
    #     # recieves data set and prodicts accuracy according to model