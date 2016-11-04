import cProfile
from DbWrapper import *
from PPRforNED import *
from Word2vecLoader import *
from models.DeepModel import DeepModel
from readers.conll_reader import *
from ModelTrainer import *
from Candidates import *

def getValue(t):
    return t[1]


_path = "/home/yotam/pythonWorkspace/deepProject"
# _path = "/home/noambox/DeepProject"

print 'Connecting to db'
wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
# wikiDB = WikipediaDbWrapper(user='noambox', password='ncTech#1', database='wiki20151002')
print 'Done!'


print "Loading iterators+stats..."
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")

ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats", fill_in=_train_stats)

#candidator = CandidatesUsingYago2(_train_stats)
#candidator.load(_path + "/data/yago2/yago.candidates")
candidator = CandidatesUsingPPRStats(ppr_stats, wikiDB)
#candidator = CandidatesUsingStatisticsObject(_train_stats)

cD = candidator.getAllCandidateSet(CoNLLIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='all'))
print "Done!"

print 'Loading embeddings...'
_w2v = Word2vecLoader(wordsFilePath=_path+"/data/word2vec/dim300vecs",
                     conceptsFilePath=_path+"/data/word2vec/dim300context_vecs")

_w2v.loadEmbeddings(conceptDict=cD)
#_w2v.randomEmbeddings(conceptDict=cD)
print 'wordEmbedding dict size: ', len(_w2v.wordEmbeddings)
print 'conceptEmbeddings dict size: ', len(_w2v.conceptEmbeddings), " wanted", len(cD)
print 'Done!'

#print 'Connecting to db'
#wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002',
#                            concept_filter=_w2v.conceptDict)
#print 'Done!'

print 'loading model'
model = DeepModel(_path + '/models/conll_model.config', w2v=_w2v, stats=ppr_stats, db=wikiDB)
predictor = model.getPredictor()
print 'Done!'

# Pre training (fine tuning model using training set)
print 'pretraining'
model.model.compile(optimizer='adagrad', loss='binary_crossentropy')
train_iter = CoNLLIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='train')
trainer = ModelTrainer(train_iter, candidator, ppr_stats, model, epochs=5, neg_sample='all')
trainer.train()
model.saveModel(_path + '/models/basic_model')
print 'Done!'

print_attn = False

tried = 0
tried_per_doc = 0

total = 0
n_docs = 0
macro_p1 = 0
gotit = 0
errors_when_no_resolved_candidates = 0
errors_due_to_unresolved_gold_sense = 0
errors_due_to_gold_sense_not_in_candidates = 0
n_candidates = 0
mps_correct = 0

mention_total = dict()
mention_correct = dict()
mention_mps_correct = dict()
sense_total = dict()
sense_correct = dict()
sense_mps_correct = dict()

n_candidates_when_wrong = 0
wrong = 0

correct_when_tried = 0
mps_when_tried = 0
print 'predicting'
f = open(_path +"/test_conll.log", "w")
test_iter = CoNLLIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='testa')
for doc in test_iter.documents():

    # add candidates
    candidator.add_candidates_to_document(doc)

    correct_per_doc = 0
    tried_per_doc = 0
    for mention in doc.mentions:
        #TODO: this is invalid?? Better not touch the gold sense.. Rather map candidates, choos the correct one, and
        #TODO: backtrack the mapping
        gold_sense_url = mention.gold_sense_url()[mention.gold_sense_url().rfind('/')+1:]
        gold_sense_id = wikiDB.resolvePage(gold_sense_url)
        mention_text = mention.mention_text().lower()
        mps = ppr_stats.getMostProbableSense(mention)

        if gold_sense_id is not None and mps == gold_sense_id:
            mps_correct += 1
        n_candidates += len(mention.candidates)
        total += 1

        if mention_text not in mention_total:
            mention_total[mention_text] = 0
            mention_mps_correct[mention_text] = 0
            mention_correct[mention_text] = 0
        if gold_sense_id not in sense_total:
            sense_total[gold_sense_id] = 0
            sense_mps_correct[gold_sense_id] = 0
            sense_correct[gold_sense_id] = 0

        mention_total[mention_text] += 1
        sense_total[gold_sense_id] += 1

        if total % 100 == 0:
            print total, " (accuracy=", str(float(gotit) / total), ")"

        correct_result = False

        if len(mention.candidates) == 0:
            errors_when_no_resolved_candidates += 1
            # 1. we could not resolve any candidates. This is usually due to the candidates being pruned for being too short
            # our only option is to get the most probable sense out of the raw candidate urls
            f.write('mention  : ' + mention.mention_text() + "\n")
            f.write("candidates: " + str(ppr_stats.getCandidateUrlsForMention(mention.mention_text())) + "\n")
            f.write("most probable sense: " + str(mps) + "\n")
            f.write("gold sense: " + str(gold_sense_url) + "\n")
            f.write("- error due to unresolved candidates\n")
            f.write("-----\n")
            f.write("\n")

        elif gold_sense_id is None:
            # 2. We could resolve at least one of the senses but not the gold sense. In this case we can't win...
            # This means that err because we pruned the gold sense! This is a very bad situation

            errors_due_to_unresolved_gold_sense += 1
            f.write('mention  : ' + mention.mention_text() + "\n")
            f.write("candidates: " + str(ppr_stats.getCandidateUrlsForMention(mention.mention_text())) + "\n")
            f.write("gold sense: " + str(gold_sense_url) + "\n")
            f.write("- error due to unresolved gold sense\n")
            f.write("-----\n")
            f.write("\n")
        elif gold_sense_id not in mention.candidates:
            # 3. we could resolve the gold sense, and candidates, but gold sense is not in the candidate list
            # we can't do nothing here..
            # this ether means the candidate list is too small (can't do anything) or we have some bug with the resolution process

            errors_due_to_gold_sense_not_in_candidates += 1
            f.write('mention  : ' + mention.mention_text() + "\n")
            f.write("candidates: " + str(ppr_stats.getCandidateUrlsForMention(mention.mention_text())) + "\n")
            f.write("gold sense: " + str(gold_sense_url) + "\n")
            f.write("- error due to golf sense not in candidates\n")
            f.write("-----\n")
            f.write("\n")
        else:
            tried += 1
            tried_per_doc += 1
            # 4. We have some candidates, and the gold sense is resolved and in the candidate list so lets test our method!
            if mps == gold_sense_id:
                mps_when_tried += 1
                mention_mps_correct[mention_text] += 1
                sense_mps_correct[gold_sense_id] += 1

            predicted = None
            predicted = predictor.predict(mention)

            # this will work only with a pointwise model at the moment
            if print_attn:
                f.write('left ctx : ' + str(mention.left_context(20)) + "\n")
                f.write('mention  : ' + mention.mention_text() + "\n")
                f.write('right ctx:' + str(mention.right_context(20)) + "\n")

                candidate_titles = [wikiDB.getPageTitle(x) for x in mention.candidates]
                f.write('candidates: ' + str(candidate_titles) + "\n")
                f.write('correct: ' + gold_sense_url + "\n")
                f.write("predicted: " + str(wikiDB.getPageTitle(predicted)) + "\n")

                attn_sum_left = None
                attn_sum_right = None
                left_context = None
                right_context = None
                k = 0
                for candidate in mention.candidates:
                    candidate_title = wikiDB.getPageTitle(candidate)
                    if candidate_title is None:
                        continue
                    ret = model.get_attn(mention, candidate, None)
                    if ret is None:
                        continue
                    k += 1
                    left_context, left_attn, right_context, right_attn = ret
                    if attn_sum_left is None:
                        attn_sum_left = left_attn
                        attn_sum_right = right_attn
                    else:
                        print attn_sum_left
                        for i in xrange(len(attn_sum_left)):
                            attn_sum_left[i] += left_attn[i]
                        for i in xrange(len(attn_sum_right)):
                            attn_sum_right[i] += right_attn[i]

                if left_context is not None:
                    for i in xrange(len(attn_sum_left)):
                        attn_sum_left[i] /= k
                    for i in xrange(len(attn_sum_right)):
                        attn_sum_right[i] /= k

                    left_context.reverse()
                    left_attn.reverse()
                    s = ''
                    for i, w in enumerate(left_context):
                        s += w + ' '
                        if left_attn[i] > 0:
                            s += '(' + str(left_attn[i]) + ') '
                    s += mention.mention_text() + ' '
                    for i, w in enumerate(right_context):
                        s += w + ' '
                        if right_attn[i] > 0:
                            s += '(' + str(right_attn[i]) + ') '
                    f.write('attention:' + s + '\n')
                f.write("-----\n")
                f.write("\n")

            if predicted == gold_sense_id:
                gotit += 1
                correct_per_doc += 1
                mention_correct[mention_text] += 1
                sense_correct[gold_sense_id] += 1
            else:
                n_candidates_when_wrong += len(mention.candidates)
                wrong += 1

                f.write('left ctx : ' + str(mention.left_context(20)) + "\n")
                f.write('mention  : ' + mention.mention_text() + "\n")
                f.write('right ctx:' + str(mention.right_context(20)) + "\n")

                candidate_titles = [wikiDB.getPageTitle(x) for x in mention.candidates]
                f.write('ppr_candidates: ' + str(ppr_stats.getCandidateUrlsForMention(mention.mention_text())))
                if len(ppr_stats.getCandidateUrlsForMention(mention.mention_text())) == 1:
                    print "111111"
                f.write('candidates: ' + str(candidate_titles) + "\n")
                f.write('correct: ' + gold_sense_url + "\n")
                f.write("predicted: " + str(wikiDB.getPageTitle(predicted)) + "\n")
                f.write("- unexplained error\n")
                f.write("-----\n")
                f.write("\n")
    # calculate macro p@1
    n_docs += 1
    macro_p1 += float(correct_per_doc) / tried_per_doc

macro_p1 /= n_docs
sense_macro_p1 = sum([float(sense_correct[x]) / sense_total[x] for x in sense_total.keys()]) / len(sense_total)
mention_macro_p1 = sum([float(mention_mps_correct[x]) / mention_total[x] for x in mention_total.keys()]) / len(mention_total)
sense_mps_macro_p1 = sum([float(sense_correct[x]) / sense_total[x] for x in sense_total.keys()]) / len(sense_total)
mention_mps_macro_p1 = sum([float(mention_mps_correct[x]) / mention_total[x] for x in mention_total.keys()]) / len(mention_total)

f.write("errors when no resolved candidates: " + str(errors_when_no_resolved_candidates) +
        "out of " + str(total) + "(" + str(float(errors_when_no_resolved_candidates) / total) + "%)\n")
f.write("errors due to unresolved gold sense: " + str(errors_due_to_unresolved_gold_sense) +
        "out of " + str(total) + "(" + str(float(errors_due_to_unresolved_gold_sense) / total) + "%)\n")
f.write("errors due to gold sense not in candidates: " + str(errors_due_to_gold_sense_not_in_candidates) +
        "out of " + str(total) + "(" + str(float(errors_due_to_gold_sense_not_in_candidates) / total) + "%)\n")
f.write("most probable sense micro accuracy: " + str(float(mps_correct) / total) + "\n")
f.write("avg. candidates per case: " + str(float(n_candidates) / total) + "\n")
f.write("avg. candidates when wrong: " + str(float(n_candidates_when_wrong) / wrong) + "\n")
f.write("\n")
f.write("tried: " + str(float(tried) / total) + " (" + str(tried) + ")\n")
f.write("mps when tried (micro: " + str(float(mps_when_tried) / tried) + "\n")
f.write("micro p@1 " + str(float(gotit) / tried) + "%\n")
f.write("macro p@1 " + str(macro_p1) + "%\n")
f.write("mention macro p@1 " + str(mention_macro_p1) + "% (mps: " + str(mention_mps_macro_p1) + "%\n")
f.write("sense macro p@1 " + str(sense_macro_p1) + "% (mps: " + str(sense_mps_macro_p1) + "%\n")
f.write("\n")
f.write("accuracy: " + str(gotit) + "out of " + str(total) + "(" + str(float(gotit) / total) + "%)\n")

f.close()

print "finished"