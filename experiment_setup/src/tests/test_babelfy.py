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

from BabelfyTester import BabelfyTester
from DbWrapper import *
from Evaluation import *
from ModelTrainer import *

##

path = ".."
print "Loading iterators+db cache..."
if(not os.path.isdir(path)):
    path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"


wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
wikiDB.cacheArticleTable()
iter_eval = WikilinksNewIterator(path+"/data/wikilinks/small/evaluation")

babelfy_model = BabelfyTester(wikiDB, path + "/data/wikilinks/babelfy")
evaluation = Evaluation(iter_eval,babelfy_model)
try:
    evaluation.evaluate()
except:
    print "nothing to do"
babelfy_model.finalizeWriter()