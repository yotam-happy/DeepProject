from DbWrapper import *
from PairwisePredict import *
from PPRforNED import *
from Word2vecLoader import *
from models.RNNPairwiseModel import *
from readers.conll_reader import *

def getValue(t):
    return t[1]

_path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats-new2")
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
print "Done!"

print 'Caching wikiDb'
wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=False)
print 'Done!'

ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats")
total_cases = 0
total_ppr_candidates = 0
total_our_candidates = 0

prediction_impossible = 0
prediction_possible = 0

gold_sense_recall = 0
ppr_gold_sense_recall = 0
ppr_candidate_mirco_recall = 0
ppr_candidate_macro_recall = 0

for i, wlink in enumerate(CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv')):

    #TODO: this is invalid. Should not touch the gold sense! Rather map candidates, choos the correct one, and
    #TODO: backtrack the mapping
    gold_sense_url = wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:]
    gold_sense_id = wikiDB.resolvePage(gold_sense_url)
    candidates_ppr = {wikiDB.resolvePage(x[x.rfind('/')+1:]): y
                  for x, y in ppr_stats.getCandidateUrlsForMention(wlink['word']).iteritems()
                  if wikiDB.resolvePage(x[x.rfind('/')+1:]) is not None}

    total_cases += 1
    if total_cases % 100 == 0:
        print "done ", total_cases

    if gold_sense_id is None:
        print "gold sense not solved: ", gold_sense_url
        prediction_impossible += 1
    else:
        if gold_sense_id in candidates_ppr:
            ppr_gold_sense_recall += 1

        prediction_possible += 1
        # 4. We have some candidates, and the gold sense is resolved and in the candidate list, can word our method!
        candidates_ours = _train_stats.getCandidatesForMention(wlink["word"], addTitleMatch=True)
        #candidates_ours = _train_stats.getCandidatesSeenWith(gold_sense_id, p=0, t=0)

        total_our_candidates += len(candidates_ours)
        total_ppr_candidates += len(candidates_ppr)

        if gold_sense_id in candidates_ours.keys():
            gold_sense_recall += 1
        else:
            print "gold sense not recalled for: ", wlink["word"], "(", gold_sense_url, ")"
            candidates_ours

        ppr_candidates_in_ours = 0
        for cand in candidates_ppr:
            if cand == gold_sense_id or cand in candidates_ours.keys():
                ppr_candidates_in_ours += 1
        ppr_candidate_mirco_recall += ppr_candidates_in_ours
        ppr_candidate_macro_recall += float(ppr_candidates_in_ours) / len(candidates_ppr)


print "total cases", total_cases, " (prediction possible: ", prediction_possible, ", ", float(prediction_possible) / total_cases, ")"
print "total ppr candidates", total_ppr_candidates, "; avg per case: ", float(total_ppr_candidates) / prediction_possible
print "total our candidates", total_our_candidates, "; avg per case: ", float(total_our_candidates) / prediction_possible
print ""
print "gold sense recall: ", float(gold_sense_recall) / prediction_possible
print "ppr gold sense recall: ", float(ppr_gold_sense_recall) / prediction_possible
print "ppr candidate micro recall: ", float(ppr_candidate_mirco_recall) / total_ppr_candidates
print "ppr candidate macro recall: ", float(ppr_candidate_macro_recall) / prediction_possible
print "finished"