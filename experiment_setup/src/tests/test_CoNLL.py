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

candidator = CandidatesUsingYago2(_train_stats)
candidator.load(_path + "/data/yago2/yago.candidates")
#candidator = CandidatesUsingPPRStats(ppr_stats, wikiDB)
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

print 'Connecting to db'
wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002',
                            concept_filter=_w2v.conceptDict)
print 'Done!'

print 'loading model'
model = DeepModel(_path + '/models/basic_model.config', w2v=_w2v, stats=_train_stats, db=wikiDB)
predictor = model.getPredictor()
print 'Done!'

# Pre training (fine tuning model using training set)
print 'pretraining'
model.model.compile(optimizer='adagrad', loss='binary_crossentropy')
train_iter = CoNLLIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='train')
trainer = ModelTrainer(train_iter, candidator, ppr_stats, model, epochs=5, neg_sample=5)
trainer.train()
model.saveModel(_path + '/models/basic_model')
print 'Done!'

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

        mps = ppr_stats.getMostProbableSense(mention)

        if gold_sense_id is not None and mps == gold_sense_id:
            mps_correct += 1

        n_candidates += len(mention.candidates)
        total += 1
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

            predicted = None
            predicted = predictor.predict(mention)

            if predicted == gold_sense_id:
                gotit += 1
                correct_per_doc += 1
            else:
                f.write('left ctx : ' + str(mention.left_context(20)) + "\n")
                f.write('mention  : ' + mention.mention_text() + "\n")
                f.write('right ctx:' + str(mention.right_context(20)) + "\n")

                candidate_titles = [wikiDB.getPageTitle(x) for x in mention.candidates]
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

f.write("errors when no resolved candidates: " + str(errors_when_no_resolved_candidates) +
        "out of " + str(total) + "(" + str(float(errors_when_no_resolved_candidates) / total) + "%)\n")
f.write("errors due to unresolved gold sense: " + str(errors_due_to_unresolved_gold_sense) +
        "out of " + str(total) + "(" + str(float(errors_due_to_unresolved_gold_sense) / total) + "%)\n")
f.write("errors due to gold sense not in candidates: " + str(errors_due_to_gold_sense_not_in_candidates) +
        "out of " + str(total) + "(" + str(float(errors_due_to_gold_sense_not_in_candidates) / total) + "%)\n")
f.write("most probable sense micro accuracy: " + str(float(mps_correct) / total) + "\n")
f.write("avg. candidates per case: " + str(float(n_candidates) / total) + "\n")
f.write("\n")
f.write("tried: " + str(float(tried) / total) + " (" + str(tried) + ")\n")
f.write("mps when tried (micro: " + str(float(mps_when_tried) / tried) + "\n")
f.write("correct when tried: micro p@1 " + str(float(gotit) / tried) + "%\n")
f.write("correct when tried: macro p@1 " + str(macro_p1) + "%\n")
f.write("\n")
f.write("accuracy: " + str(gotit) + "out of " + str(total) + "(" + str(float(gotit) / total) + "%)\n")

f.close()

print "finished"