from boto.file.key import Key
import ProjectSettings
from Candidates import *
from DbWrapper import WikipediaDbWrapper
from ModelTrainer import ModelTrainer
from PPRforNED import PPRStatistics
from WikilinksStatistics import WikilinksStatistics
from Word2vecLoader import Word2vecLoader
from readers.conll_reader import CoNLLIterator
from models import DeepModel
from GBRTmodel import GBRTModel

class Experiment():

    def __init__(self, ex_settings):
        try:
            self.path, self.pc_name = ProjectSettings.getPath()
            self.candidate_iter = ex_settings['candidate_iterator']
            self.test_iter = ex_settings['test_iterator']
            self.embedd_size = ex_settings['embedd_size']
            self.candidator_type = ex_settings['candidator_type']
            self.config = ex_settings['config_file']
            self.paramDB = {'user': ex_settings['paramDB']['user'],'db_pass':ex_settings['paramDB']['pass'],'db_name':ex_settings['paramDB']['name']}
            self.model_type = ex_settings['model_type']
            self.model_name = ex_settings['model_name']
        except KeyError:
            print 'settings not properly defined!'
            
    def prepareForTraining(self):

        wikiDB = WikipediaDbWrapper(user=self.paramDB['user'], password=self.paramDB['db_pass'], database=self.paramDB['db_name'])
        print 'prepare candidator...'
        self.ppr_stats = PPRStatistics(None, self.path + "/data/PPRforNED/ppr_stats", fill_in=self.train_stats)
        self.train_stats = WikilinksStatistics(None, load_from_file_path= self.path + "/data/intralinks/train-stats")
        self.candidator = self.switchCandidator(self.candidator_type, _db = wikiDB)
        if self.candidator_type is 'yago':
            self.candidator.load(self.path + "/data/yago2/yago.candidates")

        cD = self.candidator.getAllCandidateSet(self.candidate_iter)
        print "Done!"

        print 'Loading embeddings...'
        _w2v = Word2vecLoader(wordsFilePath=self.path + "/data/word2vec/" + self.embedd_size + "vecs",
                              conceptsFilePath=self.path + "/data/word2vec/" + self.embedd_size + "context_vecs")

        _w2v.loadEmbeddings(conceptDict=cD)
        # _w2v.randomEmbeddings(conceptDict=cD)
        print 'wordEmbedding dict size: ', len(_w2v.wordEmbeddings)
        print 'conceptEmbeddings dict size: ', len(_w2v.conceptEmbeddings), " wanted", len(cD)
        print 'Done!'

        print 'Connecting to db'
        self.wikiDB = WikipediaDbWrapper(user=self.paramDB['user'], password=self.paramDB['db_pass'], database=self.paramDB['db_name'],
                                         concept_filter=_w2v.conceptDict)
        print 'Done!'

        print 'loading model'
        self.model = self.switchModel(self, self.model_type, _w2v )
        self.predictor = self.model.getPredictor()
        print 'Done!'

    def trainTheModel(self,epochs=5, neg_sample=1, save_flag = True):
        print 'beging training...'
        trainer = ModelTrainer(self.candidate_iter, self.candidator, self.ppr_stats, self.model,epochs=5, neg_sample=1)
        trainer.train()
        if save_flag:
            self.model.saveModel(self.path + '/models/' + self.model_name)
        print 'Done!'

    def predictModel(self):

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
        f = open(self.path + "/test_conll" + self.model_name +".log", "w")
        test_iter = self.test_iter
        for doc in test_iter.documents():

            # add candidates
            self.candidator.add_candidates_to_document(doc)

            correct_per_doc = 0
            tried_per_doc = 0
            for mention in doc.mentions:
                # TODO: this is invalid?? Better not touch the gold sense.. Rather map candidates, choos the correct one, and
                # TODO: backtrack the mapping
                gold_sense_url = mention.gold_sense_url()[mention.gold_sense_url().rfind('/') + 1:]
                gold_sense_id = wikiDB.resolvePage(gold_sense_url)

                mps = self.ppr_stats.getMostProbableSense(mention)

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
                    f.write("candidates: " + str(self.ppr_stats.getCandidateUrlsForMention(mention.mention_text())) + "\n")
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
                    f.write("candidates: " + str(self.ppr_stats.getCandidateUrlsForMention(mention.mention_text())) + "\n")
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
                    f.write("candidates: " + str(self.ppr_stats.getCandidateUrlsForMention(mention.mention_text())) + "\n")
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
                    predicted = self.predictor.predict(mention)

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

        pass

    def switchCandidator(self, candidator, _db = None):

        # train on wikipedia intra-links corpus
        try:
            return {
                'yago': CandidatesUsingYago2(self.train_stats),
                'ppr': CandidatesUsingPPRStats(self.ppr_stats, _db),
                'wikistats':CandidatesUsingStatisticsObject(self.train_stats),
            }[candidator]
        except KeyError:
            raise ValueError('bad candidator type')

    def switchModel(self, model_type, _w2v ):
        try:
            return{
                'deep_model': DeepModel(self.path + '/models/' + self.config, w2v=_w2v, stats= self._train_stats, db=self.wikiDB),
                'gbrt': GBRTModel(self.path + '/models/' + self.config),
            }[model_type]
        except KeyError:
            raise ValueError('bad model type')