"""
Mannually execute each class dependency to work with the project without compiling
Best and easiest way to debug code in python. You can actully view the whole workspace

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
import nltk
from BabelfyTester import BabelfyTester
from DbWrapper import *

##

path = "C:\\repo\\DeepProject"
print "Loading iterators+stats..."
if(not os.path.isdir(path)):
    path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\small\\evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
print "Done!"

wikiDB = WikipediaDbWrapper(user='root', password='rockon123', database='wikiprep-esa-en20151002')
babelfy_model = BabelfyTester(wikiDB)
evaluation = Evaluation(iter_eval,babelfy_model)
evaluation.evaluate()