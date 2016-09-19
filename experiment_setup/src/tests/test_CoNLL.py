from DbWrapper import *
from KnockoutModel import *
from PPRforNED import *
from Word2vecLoader import *
from models.RNNPairwiseModel import *
from tests.ConllReader import *


def getValue(t):
    return t[1]


_path = "/home/yotam/pythonWorkspace/deepProject"
pc_name = 'yotam'
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"
    pc_name = 'noam'

# train on wikipedia intra-links corpus
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_iter_train = WikilinksNewIterator(_path+"/data/intralinks/train-filtered")
#_iter_eval = WikilinksNewIterator(_path+"/data/intralinks/test-filtered")

print "Loading iterators+stats..."
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
_iter_train = WikilinksNewIterator(_path+"/data/wikilinks/all/train")
_iter_eval = WikilinksNewIterator(_path+"/data/wikilinks/all/evaluation")
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
if pc_name == 'yotam':
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=True)
elif pc_name == 'noam':
    wikiDB = WikipediaDbWrapper(user='root', password='ncTech#1', database='wikiprep-esa-en20151002', cache=True)
print 'Done!'

print 'loading model'
model_path = _path + '/models/small.0.out'
_pairwise_model = RNNPairwiseModel(_w2v, _train_stats, dropout=0.1)
_pairwise_model.loadModel(model_path)
knockout_model = KnockoutModel(_pairwise_model, None)
print 'Done!'

ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats")
total = 0
gotit = 0
errors_when_no_resolved_candidates = 0
errors_due_to_unresolved_gold_sense = 0
errors_due_to_gold_sense_not_in_candidates = 0
mps_correct = 0

for i, wlink in enumerate(CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv')):

    #TODO: this is invalid. Should not touch the gold sense! Rather map candidates, choos the correct one, and
    #TODO: backtrack the mapping
    gold_sense_url = wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:]
    gold_sense_id = wikiDB.resolvePage(gold_sense_url)
    candidates = {wikiDB.resolvePage(x[x.rfind('/')+1:]): y
                  for x, y in ppr_stats.getCandidateUrlsForMention(wlink['word']).iteritems()
                  if wikiDB.resolvePage(x[x.rfind('/')+1:]) is not None}
    candidates_to_print = {x[x.rfind('/')+1:]: y
                  for x, y in ppr_stats.getCandidateUrlsForMention(wlink['word']).iteritems()
                  if wikiDB.resolvePage(x[x.rfind('/')+1:]) is not None}

    mps = ppr_stats.getMostProbableSense(wlink['word'])
    mps = mps[mps.rfind('/') + 1:]
    mps = mps.encode('utf8')

    if mps == gold_sense_url:
        mps_correct += 1

    total += 1

    correct_result = False

    if len(candidates) == 0:
        # 1. we could not resolve any candidates. This is usually due to the candidates being pruned for being too short
        # our only option is to get the most probable sense out of the raw candidate urls

        if mps == gold_sense_url:
            correct_result = True
        else:
            errors_when_no_resolved_candidates += 1
            print 'mention  : ', wlink['word']
            print "candidates: ", ppr_stats.getCandidateUrlsForMention(wlink['word'])
            print "most probable sense: ", mps
            print "gold sense: ", gold_sense_url
            print "- error due to unresolved candidates: "
            print "-----"
            print ""

    elif gold_sense_id is None:
        # 2. We could resolve at least one of the senses but not the gold sense. In this case we can't win...
        # This means that err because we pruned the gold sense! This is a very bad situation

        errors_due_to_unresolved_gold_sense += 1
        print 'mention  : ', wlink['word']
        print "candidates: ", ppr_stats.getCandidateUrlsForMention(wlink['word'])
        print "resolved candidates: ", candidates_to_print
        print "gold sense: ", gold_sense_url
        print "- error due to unresolved gold sense"
        print "-----"
        print ""
    elif gold_sense_id not in candidates:
        # 3. we could resolve the gold sense, and candidates, but gold sense is not in the candidate list
        # we can't do nothing here..
        # this ether means the candidate list is too small (can't do anything) or we have some bug with the resolution process

        errors_due_to_gold_sense_not_in_candidates += 1
        print 'mention  : ', wlink['word']
        print "candidates: ", ppr_stats.getCandidateUrlsForMention(wlink['word'])
        print "resolved candidates: ", candidates_to_print
        print "gold sense: ", gold_sense_url
        print "- error due to gold sense not in candidates"
        print "-----"
        print ""
    else:
        # 4. We have some candidates, and the gold sense is resolved and in the candidate list so lets test our method!

        predicted = None
        if len(candidates) == 1:
            predicted = [x for x in candidates][0]
        else:
            predicted = knockout_model.predict(wlink, candidates)

        if predicted == gold_sense_id:
            correct_result = True
        else:
            print 'left ctx : ', wlink['left_context_text']
            print 'mention  : ', wlink['word']
            print 'right ctx:', wlink['right_context_text']

            cands_title = ppr_stats.getCandidateUrlsForMention(wlink['word'])
            cands_title = [(x[0][x[0].rfind('/')+1:],x[1]) for x in cands_title.items()]
            cands_title.sort(key=getValue, reverse=True)

            candidates_ppr = [(wikiDB.getArticleTitleById(x), y) for x, y in candidates.items()]
            candidates_ppr.sort(key=getValue, reverse=True)
            candidates_wl = [(wikiDB.getArticleTitleById(x), y) for x, y in _train_stats.getCandidatesForMention(wlink['word']).items()]
            candidates_wl.sort(key=getValue, reverse=True)
            candidates_seenWith = [(wikiDB.getArticleTitleById(x), y) for x, y in _train_stats.getCandidatesForMention(wlink['word']).items()]
            candidates_seenWith.sort(key=getValue, reverse=True)

            print "pprforned: ", candidates_ppr
            print "wlinks   : ", candidates_wl
            print "gold sense seen with: ", candidates_seenWith

            print 'candidates: ', cands_title
            print 'correct: ', wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:]
            print "predicted: ", wikiDB.getArticleTitleById(predicted)
            print "- unexplained error"
            print "-----"
            print ""

    if correct_result:
        gotit += 1


print "errors when no resolved candidates: ", errors_when_no_resolved_candidates, "out of ", total, "(", float(errors_when_no_resolved_candidates) / total, "%)"
print "errors due to unresolved gold sense: ", errors_due_to_unresolved_gold_sense, "out of ", total, "(", float(errors_due_to_unresolved_gold_sense) / total, "%)"
print "errors due to gold sense not in candidates: ", errors_due_to_gold_sense_not_in_candidates, "out of ", total, "(", float(errors_due_to_gold_sense_not_in_candidates) / total, "%)"
print "most probable sense correct: ", float(mps_correct) / total
print ""
print "accuracy: ", gotit, "out of ", total, "(", float(gotit) / total, "%)"
print "p@1 score: ", float(gotit) / (total-errors_due_to_gold_sense_not_in_candidates) , "%"
