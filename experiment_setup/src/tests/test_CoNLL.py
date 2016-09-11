from DbWrapper import *
from KnockoutModel import *
from PPRforNED import *
from Word2vecLoader import *
from models.RNNPairwiseModel import *
from tests.ConllReader import *


def getValue(t):
    return t[1]


_path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_iter_train = WikilinksNewIterator(_path+"/data/intralinks/train-filtered")
#_iter_eval = WikilinksNewIterator(_path+"/data/intralinks/test-filtered")

_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
_iter_train = WikilinksNewIterator(_path+"/data/wikilinks/filtered/train")
_iter_eval = WikilinksNewIterator(_path+"/data/wikilinks/filtered/evaluation")
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

print 'Caching wikiDb'
wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=True)
print 'Done!'

print 'loading model'
model_path = _path + '/models/small.4.out'
_pairwise_model = RNNPairwiseModel(_w2v, _train_stats, dropout=0.1)
_pairwise_model.loadModel(model_path)
knockout_model = KnockoutModel(_pairwise_model, None)
print 'Done!'

ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats")
total = 0
gotit = 0
possible = 0
gold_sense_in_train = 0
for i, wlink in enumerate(CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv')):

    #TODO: this is invalid. Should not touch the gold sense! Rather map candidates, choos the correct one, and
    #TODO: backtrack the mapping
    gold_sense = wikiDB.resolvePage(wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:])
    candidates = {wikiDB.resolvePage(x[x.rfind('/')+1:]): y
                  for x, y in ppr_stats.getCandidateUrlsForMention(wlink['word']).iteritems()
                  if wikiDB.resolvePage(x[x.rfind('/')+1:]) is not None}

    #fake candidate priors by incounts

    predicted = None
    if len(candidates) == 0:
        predicted = [x for x in ppr_stats.getCandidatesForMention(wlink['word']).keys()][0]
    elif len(candidates) == 1:
        predicted = [x for x in candidates][0]
    else:
        predicted = knockout_model.predict(wlink, candidates)
        print 'left ctx : ', wlink['left_context_text']
        print 'mention  : ', wlink['word']
        print 'right ctx:', wlink['right_context_text']

        cands_title = ppr_stats.getCandidateUrlsForMention(wlink['word'])
        cands_title = [(x[0][x[0].rfind('/')+1:],x[1]) for x in cands_title.items()]
        cands_title.sort(key=getValue, reverse=True)

        candidates_ppr = [x for x in candidates.items()]
        candidates_ppr.sort(key=getValue, reverse=True)
        candidates_wl = [x for x in _train_stats.getCandidatesForMention(wlink['word']).items()]
        candidates_wl.sort(key=getValue, reverse=True)

        print "pprforned: ", candidates_ppr
        print "wlinks   : ", candidates_wl

        print 'candidates: ', cands_title
        print 'correct: ', wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:]
        print "predicted: ", wikiDB.getArticleTitleById(predicted)
        if predicted is not None and predicted == gold_sense:
            print "hi!!!!!!"
        else:
            print "no~~~~~~"

    total += 1
    if gold_sense in candidates:
        possible += 1
    #if gold_sense in _train_stats.getCandidatesForMention(wlink['word']):
    #    gold_sense_in_train += 1
    if predicted is not None and predicted == gold_sense:
        gotit += 1

print "accuracy: ", gotit, "out of ", total, "(", float(gotit) / total, "%)"
print "accuracy where tried: ", gotit, "out of ", possible, "(", float(gotit) / possible, "%)"
