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
print "Loading iterators+stats..."
if(not os.path.isdir(_path)):
    _path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

# train on wikipedia intra-links corpus
_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/intralinks/train-stats")
#_train_stats = WikilinksStatistics(None, load_from_file_path=_path+"/data/wikilinks/train-stats")

ppr_stats = PPRStatistics(None, _path+"/data/PPRforNED/ppr_stats")
cD = ppr_stats.conceptCounts

print "Done!"

print 'Loading embeddings...'
_w2v = Word2vecLoader(wordsFilePath=_path+"/data/word2vec/new/dim300vecs",
                     conceptsFilePath=_path+"/data/word2vec/new/dim300context_vecs")

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

candidator = CandidatesUsingPPRStats(ppr_stats, wikiDB)

# Pre training (fine tuning model using training set)
print 'pretraining'
model.model.compile(optimizer='adagrad', loss='binary_crossentropy')
train_iter = CoNLLIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='train')
trainer = ModelTrainer(train_iter, candidator, ppr_stats, model, epochs=1, neg_sample=5)
trainer.train()
model.saveModel(_path + '/models/basic_model')
print 'Done!'

not_tried = 0

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
print 'predicting'
f = open(_path +"/test_conll.log", "w")
test_iter = CoNLLIterator(_path+'/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', split='testa')
for doc in test_iter.documents():

    # add candidates
    candidator.add_candidates_to_document(doc)

    for mention in doc.mentions:
        #TODO: this is invalid?? Better not touch the gold sense.. Rather map candidates, choos the correct one, and
        #TODO: backtrack the mapping
        gold_sense_url = mention.gold_sense_url()[mention.gold_sense_url().rfind('/')+1:]
        gold_sense_id = wikiDB.resolvePage(gold_sense_url)

        mps = ppr_stats.getMostProbableSense(mention.mention_text())
        mps = mps[mps.rfind('/') + 1:]
        mps = mps.encode('utf8')

        if mps == gold_sense_url:
            mps_correct += 1

        total += 1
        if total % 100 == 0:
            print total, " (accuracy=", str(float(gotit) / total), ")"

        correct_result = False

        if len(mention.candidates) == 0:
            # 1. we could not resolve any candidates. This is usually due to the candidates being pruned for being too short
            # our only option is to get the most probable sense out of the raw candidate urls

            not_tried += 1
            if mps == gold_sense_url:
                correct_result = True
                correct_result_by_title = True
            else:
                errors_when_no_resolved_candidates += 1
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
            # 4. We have some candidates, and the gold sense is resolved and in the candidate list so lets test our method!
            if mps == gold_sense_url:
                mps_when_tried += 1

            predicted = None
            predicted = predictor.predict(mention)

            if predicted == gold_sense_id:
                correct_result = True
                correct_when_tried += 1
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
f.write("correct when tried (p@1): " + str(float(correct_when_tried) / total) + "%; mps when tried: " + str(float(mps_when_tried) / total) + "\n")
f.write("not tried: " + str(not_tried) + "\n")
f.write("\n")
f.write("accuracy: " + str(gotit) + "out of " + str(total) + "(" + str(float(gotit) / total) + "%)\n")

f.close()

print "finished"