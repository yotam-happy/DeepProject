
# loading w2v and stats
## The cell seperator

from PyQt4 import QtGui

import deepesa_ui
from ModelTrainer import *

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

##
print 'Loading embeddings...'
w2v = Word2vecLoader(wordsFilePath=path+"/data/word2vec/dim300vecs",
                     conceptsFilePath=path+"/data/word2vec/dim300context_vecs")
wD = train_stats.contextDictionary
cD = train_stats.conceptCounts
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(w2v.wordEmbeddings)
print 'conceptEmbeddings dict size: ',len(w2v.conceptEmbeddings)
print 'Done!'

## running the ui with w2v and stats with its __main__()

w2v_stats_dic = {'w2v': w2v, 'stats': train_stats, 'iter':iter_eval}
app = QtGui.QApplication(sys.argv)
MainWindow = QtGui.QMainWindow()
ui = deepesa_ui.Ui_MainWindow(w2v_stats_dic)
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())
##
w2v_stats_dic = {'w2v': None, 'stats': None, 'iter':iter_eval}

##
pairwise_model = RNNPairwiseModel(w2v)
model_file = pairwise_model.loadModel("C:\\Users\\Noam\\Documents\\GitHub\\DeepProject\\models\\rnn_gru.1")
model = KnockoutModel(pairwise_model=model_file, stats=train_stats)
