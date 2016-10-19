"""
This scrip is used for creating files of features and labeling instead
of using a generator
"""
# from blaze.compute.tests.test_pyfunc import inc

from DbWrapper import WikipediaDbWrapper
from PPRforNED import PPRStatistics
from WikilinksStatistics import WikilinksStatistics
from readers.conll_reader import CoNLLWikilinkIterator
import os
from FeatureGenerator import FeatureGenerator
import numpy as np
import pandas as pd
from models.RNNPairwiseModel import RNNPairwiseModel
from KnockoutModel import *
from Word2vecLoader import *

_path = "/home/yotam/pythonWorkspace/deepProject"
pc_name = 'yotam'
if not os.path.isdir(_path):
    pc_name = 'euterpe'
    _path = "/home/noambox/DeepProject"
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"
    pc_name = 'noam'

print 'Caching wikiDb'
if pc_name == 'yotam':
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=False)
elif pc_name == 'euterpe':
    wikiDB = WikipediaDbWrapper(user='noambox', password='ncTech#1', database='wiki20151002', cache=False)
elif pc_name == 'noam':
    wikiDB = WikipediaDbWrapper(user='root', password='ncTech#1', database='wikiprep-esa-en20151002', cache=False)
print 'Done!'

print 'Loading statistics...'
ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats")
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
print 'Done!'

##
print 'Loading embeddings...'
_w2v = Word2vecLoader(wordsFilePath=_path+"/data/word2vec/dim300vecs",
                     conceptsFilePath=_path+"/data/word2vec/dim300context_vecs")
wD = _train_stats.contextDictionary
cD = _train_stats.conceptCounts
_w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
print 'wordEmbedding dict size: ',len(_w2v.wordEmbeddings), " wanted: ", len(wD)
print 'conceptEmbeddings dict size: ',len(_w2v.conceptEmbeddings), " wanted", len(cD)
print 'Done!'
##

print 'loading model'
model_path = _path + '/models/small.0.out'
_feature_generator = FeatureGenerator(entity_features={'log_prior', 'cond_prior'}, stats=_train_stats, feature_consistancy = False)
_pairwise_model = RNNPairwiseModel(_w2v, dropout=0.5, feature_generator=_feature_generator)
_pairwise_model.loadModel(model_path)
_knockout_model = KnockoutModel(_pairwise_model, None)
print 'Done!'

##

fg = FeatureGenerator(mention_features=  {'yamadas_base'}, stats =  ppr_stats, db = wikiDB)
# fg = FeatureGenerator(mention_features=  {'yamadas_base','rnn_model_feature'}, stats =  ppr_stats, db = wikiDB, knockout_model=_knockout_model,  feature_consistancy = False)
yamada_base_features = pd.DataFrame()
print 'Start creating base DS!'

train_iter = CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='train')
testa_iter = CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='testa', includeUnresolved=False)
testb_iter = CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='testb', includeUnresolved=False)

count = 0
for wlink_list in testb_iter.wikilinks(all_mentions_per_doc = True):
    for feature_vec in fg.getMentionListFeatures(wlink_list):
        yamada_base_features = pd.concat([ yamada_base_features, feature_vec ])
    # yamada_base_features = pd.concat([yamada_base_features, wlink_features.next()])
    count += 1
    print '\t iter number: \t:', count
    # if count > 5: break
print 'done iterating!'

# save

# yamada_base_features.to_pickle(_path +"/data/CoNLL/train_data_base_conll")
yamada_base_features.to_pickle(_path +"/data/CoNLL/testb_data_base_conll")


##
# fixing the ppr_stats problem locally (there is a fixme in the PPRforNED class)
# count = 0
ppr_stats.conceptCounts = dict()
for mention, concepts in ppr_stats.mentionLinks.iteritems():
    for key,val in concepts.iteritems():
        ppr_stats.conceptCounts[key] = ppr_stats.conceptCounts.get(key, 0) + val
print 'Done!'
##
