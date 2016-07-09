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
from VanilllaNNPairwiseModel import *
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
print 'Starts model evluation\nStarts loading files...'
os.chdir("C:\\repo\\DeepProject") # TODO: Yotam, you need to change this in order to work with this file
path = "C:\\repo\\DeepProject"
train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")

iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))

print 'Load embeddings...'
w2v = Word2vecLoader(wordsFilePath=path+"\\data\\word2vec\\dim300vecs",
                     conceptsFilePath=path+"\\data\\word2vec\\dim300context_vecs")
wD = train_stats.mentionLinks
cD = train_stats.conceptCounts
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'done'

## TRAIN DEBUGGING CELL
print 'Training...'
pairwise_model = VanillaNNPairwiseModel(w2v)
knockout_model = KnockoutModel(pairwise_model,train_stats)
trainer = ModelTrainer(iter_train, train_stats, pairwise_model)
trainer.train()
