from DbWrapper import *
from KnockoutModel import *
from PPRforNED import *
from Word2vecLoader import *
from models.RNNPairwiseModel import *
from tests.ConllReader import *
from ModelTrainer import *
from FeatureGenerator import *

def getValue(t):
    return t[1]


_path = "/home/yotam/pythonWorkspace/deepProject"
print "Loading iterators+stats..."
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")
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
wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=False)
print 'Done!'

ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats")

print 'loading model'
model_path = _path + '/models/small.0.out'
_feature_generator = FeatureGenerator(entity_features={'log_prior', 'cond_prior'}, stats=_train_stats)
_pairwise_model = RNNPairwiseModel(_w2v, dropout=0.5, feature_generator=_feature_generator)
_pairwise_model.loadModel(model_path)
knockout_model = KnockoutModel(_pairwise_model, None)
print 'Done!'


# Pre training (fine tuning model using training set)
#print 'pretraining'
#_pairwise_model.model.compile(optimizer='adagrad', loss='binary_crossentropy')
#train_iter = CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='train')
#trainer = ModelTrainer(train_iter, ppr_stats, _pairwise_model, epochs=5)
#trainer.train()
#print 'Done!'

total = 0
gotit = 0
errors_when_no_resolved_candidates = 0
errors_due_to_unresolved_gold_sense = 0
errors_due_to_gold_sense_not_in_candidates = 0

mps_correct = 0
correct_when_mps_wrong = 0
wrong_when_mps_correct = 0

correct_when_tried = 0
mps_when_tried = 0

f = open(_path +"/test_conll.log", "w")
test_iter = CoNLLWikilinkIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv')
for i, wlink in enumerate(test_iter.wikilinks()):

    #TODO: this is invalid. Should not touch the gold sense! Rather map candidates, choos the correct one, and
    #TODO: backtrack the mapping
    gold_sense_url = wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:]
    gold_sense_id = wikiDB.resolvePage(gold_sense_url)
#    candidates = {wikiDB.resolvePage(x[x.rfind('/')+1:]): y
#                  for x, y in ppr_stats.getCandidateUrlsForMention(wlink['word']).iteritems()
#                  if wikiDB.resolvePage(x[x.rfind('/')+1:]) is not None}
#    candidates_to_print = {x[x.rfind('/')+1:]: y
#                  for x, y in ppr_stats.getCandidateUrlsForMention(wlink['word']).iteritems()
#                  if wikiDB.resolvePage(x[x.rfind('/')+1:]) is not None}
    candidates = _train_stats.getCandidatesForMention(wlink['word'])
    candidates_to_print = None

    mps = ppr_stats.getMostProbableSense(wlink['word'])
    mps = mps[mps.rfind('/') + 1:]
    mps = mps.encode('utf8')

    if mps == gold_sense_url:
        mps_correct += 1

    total += 1
    if total % 100 == 0:
        print total, " (accuracy=", str(float(gotit) / total), ")"

    correct_result = False

    if len(candidates) == 0:
        # 1. we could not resolve any candidates. This is usually due to the candidates being pruned for being too short
        # our only option is to get the most probable sense out of the raw candidate urls

        if mps == gold_sense_url:
            correct_result = True
        else:
            errors_when_no_resolved_candidates += 1
            f.write('mention  : ' + str(wlink['word']) + "\n")
            f.write("candidates: " + str(ppr_stats.getCandidateUrlsForMention(wlink['word'])) + "\n")
            f.write("most probable sense: " + str(mps) + "\n")
            f.write("gold sense: " + str(gold_sense_url) + "\n")
            f.write("- error due to unresolved candidates\n")
            f.write("-----\n")
            f.write("\n")

    elif gold_sense_id is None:
        # 2. We could resolve at least one of the senses but not the gold sense. In this case we can't win...
        # This means that err because we pruned the gold sense! This is a very bad situation

        errors_due_to_unresolved_gold_sense += 1
        f.write('mention  : ' + str(wlink['word']) + "\n")
        f.write("candidates: " + str(ppr_stats.getCandidateUrlsForMention(wlink['word'])) + "\n")
        f.write("resolved candidates: " + str(candidates_to_print) + "\n")
        f.write("gold sense: " + str(gold_sense_url) + "\n")
        f.write("- error due to unresolved gold sense\n")
        f.write("-----\n")
        f.write("\n")
    elif gold_sense_id not in candidates:
        # 3. we could resolve the gold sense, and candidates, but gold sense is not in the candidate list
        # we can't do nothing here..
        # this ether means the candidate list is too small (can't do anything) or we have some bug with the resolution process

        errors_due_to_gold_sense_not_in_candidates += 1
        f.write('mention  : ' + str(wlink['word']) + "\n")
        f.write("candidates: " + str(ppr_stats.getCandidateUrlsForMention(wlink['word'])) + "\n")
        f.write("resolved candidates: " + str(candidates_to_print) + "\n")
        f.write("gold sense: " + str(gold_sense_url) + "\n")
        f.write("- error due to golf sense not in candidates\n")
        f.write("-----\n")
        f.write("\n")
    else:
        # 4. We have some candidates, and the gold sense is resolved and in the candidate list so lets test our method!
        if mps == gold_sense_url:
            mps_when_tried += 1

        predicted = None
        if len(candidates) == 1:
            predicted = [x for x in candidates][0]
        else:
            predicted = knockout_model.predict(wlink, candidates)
            #predicted = knockout_model.predict2(wlink, candidates)

        if predicted == gold_sense_id:
            correct_result = True
            correct_when_tried += 1
        else:
            f.write('left ctx : ' + str(wlink['left_context_text']) + "\n")
            f.write('mention  : ' + str(wlink['word']) + "\n")
            f.write('right ctx:' + str(wlink['right_context_text']) + "\n")

            cands_title = ppr_stats.getCandidateUrlsForMention(wlink['word'])
            cands_title = [(x[0][x[0].rfind('/')+1:],x[1]) for x in cands_title.items()]
            cands_title.sort(key=getValue, reverse=True)

            candidates_ppr = [(wikiDB.getArticleTitleById(x), y) for x, y in candidates.items()]
            candidates_ppr.sort(key=getValue, reverse=True)
            candidates_wl = [(wikiDB.getArticleTitleById(x), y) for x, y in _train_stats.getCandidatesForMention(wlink['word']).items()]
            candidates_wl.sort(key=getValue, reverse=True)
#            candidates_seenWith = [(wikiDB.getArticleTitleById(x), y) for x, y in _train_stats.getCandidatesForMention(wlink['word']).items()]
#            candidates_seenWith.sort(key=getValue, reverse=True)

            f.write("pprforned: " + str(candidates_ppr) + "\n")
            f.write("wlinks   : " + str(candidates_wl) + "\n")
#            f.write("gold sense seen with: " + str(candidates_seenWith) + "\n")

            f.write('candidates: ' + str(cands_title) + "\n")
            f.write('correct: ' + str(wlink['wikiurl'][wlink['wikiurl'].rfind('/')+1:]) + "\n")
            f.write("predicted: " + str(wikiDB.getArticleTitleById(predicted)) + "\n")
            f.write("- unexplained error\n")
            f.write("-----\n")
            f.write("\n")

    if correct_result:
        gotit += 1
    if correct_result and mps != gold_sense_url:
        correct_when_mps_wrong += 1
    if not correct_result and mps == gold_sense_url:
        wrong_when_mps_correct += 1

f.write("errors when no resolved candidates: " + str(errors_when_no_resolved_candidates) +
        "out of " + str(total) + "(" + str(float(errors_when_no_resolved_candidates) / total) + "%)\n")
f.write("errors due to unresolved gold sense: " + str(errors_due_to_unresolved_gold_sense) +
        "out of " + str(total) + "(" + str(float(errors_due_to_unresolved_gold_sense) / total) + "%)\n")
f.write("errors due to gold sense not in candidates: " + str(errors_due_to_gold_sense_not_in_candidates) +
        "out of " + str(total) + "(" + str(float(errors_due_to_gold_sense_not_in_candidates) / total) + "%)\n")
f.write("most probable sense correct: " + str(float(mps_correct) / total) + "\n")
f.write("correct when  mps wrong: " + str(float(correct_when_mps_wrong) / total) + "\n")
f.write("wrong when mps correct: " + str(float(wrong_when_mps_correct) / total) + "\n")
f.write("correct when tried: " + str(float(correct_when_tried) / total) + "%; mps when tried: " + str(float(mps_when_tried) / total) + "\n")
f.write("\n")
f.write("accuracy: " + str(gotit) + "out of " + str(total) + "(" + str(float(gotit) / total) + "%)\n")
f.close()

print "finished"