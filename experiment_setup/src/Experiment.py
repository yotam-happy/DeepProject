import ProjectSettings
from Candidates import *
from DbWrapper import WikipediaDbWrapper
from DbWrapperCached import *
from ModelTrainer import ModelTrainer
from PPRforNED import PPRStatistics
from WikilinksStatistics import WikilinksStatistics
from Word2vecLoader import Word2vecLoader
from readers.conll_reader import CoNLLIterator
from GBRTmodel import GBRTModel
from WikilinksIterator import *
from Evaluation import *
from models.DeepModel import *
import cProfile

class Experiment:
    def __init__(self, config):
        self.path, self.pc_name = ProjectSettings.getPath()

        if type(config) == str:
            with open(self.path + config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config

        self.db = self.connect_db(self._config['db'])

        self.iterators = dict()
        for x in self._config["iterators"]:
            self.iterators[x["name"]] = self.switch_iterator(x)

        self.stats = dict()
        for x in self._config["stats"]:
            self.stats[x["name"]] = self.load_stats(x)

        self.candidator = self.switch_candidator(self._config['candidator'])
        self.w2v = self.load_w2v(self._config['w2v']) if 'w2v' in self._config else None
        self.model = self.switch_model(self._config['model'])

        self.trained_mentions = None

        # TODO: somehow fix this serious hack??
        if hasattr(self.model._feature_generator, 'yamada_txt_to_embd') \
                and self.model._feature_generator.yamada_txt_to_embd is not None:
            self.candidator._filter = self.model._feature_generator.yamada_txt_to_embd

    def switch_model(self, config):
        dmodel = None
        if 'distance_model' in config:
            dmodel = self.switch_model(config['distance_model'])

        print "loading model..."
        if config["type"] == 'deep_model':
            return DeepModel(self.path + config['config_path'],
                             w2v=self.w2v,
                             stats=self.stats[config['stats']],
                             db=self.db,
                             load_path=self.path + str(config['load_path']) if 'load_path' in config else None,
                             dmodel=dmodel)
        elif config["type"] == 'gbrt':
            return GBRTModel(self.path + config['config_path'],
                             db=self.db,
                             stats=self.stats[config['stats']],
                             dmodel=dmodel,
                             load_path=self.path + str(config['load_path']) if 'load_path' in config else None,
                             w2v=self.w2v)
        else:
            raise "Config error"

    def load_w2v(self, config):
        print "loading w2v..."
        w2v = Word2vecLoader(wordsFilePath=self.path + config['words_path'],
                             conceptsFilePath=self.path + config['concepts_path'])
        concept_filter = self.switch_concept_filter(config['concept_filter']) if 'concept_filter' in config else None
        if 'random' in config and config['random']:
            w2v.randomEmbeddings(conceptDict=concept_filter)
        else:
            w2v.loadEmbeddings(conceptDict=concept_filter)
        return w2v

    def switch_concept_filter(self, config):
        if config['src'] == 'by_iter':
            return self.candidator.getAllCandidateSet(self.iterators[config['iterator']])
        elif config['src'] == 'by_stats':
            return {int(x) for x in self.stats[config['stats']].conceptCounts}
        else:
            raise "Config error"

    def load_stats(self, config):
        print "loading statistics:", config["name"]
        if config['src'] == 'stats_file':
            return WikilinksStatistics(None, load_from_file_path=self.path + config['path'])
        elif config['src'] == 'ppr':
            return PPRStatistics(None, self.path + config['path'],
                                 fill_in=self.stats[config['fillin']] if 'fillin' in config else None)
        else:
            raise "Config error"

    def connect_db(self, config):
        print "connecting to db..."
        if 'cached' in config and config['cached']:
            print "load cache"
            db = WikipediaDbWrapperCached()
            db.load(self.path + config['path'])
            return db
        else:
            return WikipediaDbWrapper(user=config['user'],
                                      password=config['password'],
                                      database=config['database'])

    def switch_iterator(self, config):
        if config['dataset'] == 'conll':
            return CoNLLIterator(self.path + '/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', self.db, split=config['split'])
        elif config['dataset'] == 'from_json':
            return WikilinksNewIterator(self.path + config['path'])
        else:
            raise "Config error"

    def switch_candidator(self, config):
        if config['type'] == 'yago':
            return CandidatesUsingYago2(self.stats[config['more_stats']])
        elif config['type'] == 'ppr':
            return CandidatesUsingPPRStats(self.stats[config['stats']], self.db)
        elif config['type'] == 'stats':
            return CandidatesUsingStatisticsObject(self.stats[config['stats']])
        else:
            raise "Config error"

    def train(self, config=None, model_name=None):
        if config is None:
            config = self._config["training"]
        if 'train' in config and not config['train']:
            return

        print 'beging training...'
        trainer = ModelTrainer(self.iterators[config['iterator']],
                               self.candidator,
                               self.stats[config['stats']],
                               self.model,
                               epochs=config['epochs'],
                               neg_sample=config['neg_samples'],
                               neg_sample_uniform=config['neg_sample_uniform'],
                               neg_sample_all_senses_prob=config['neg_sample_all_senses_prob'],
                               sampling=config['sampling'] if 'sampling' in config else None,
                               subsampling=config['subsampling'] if 'subsampling' in config else None)
        self.trained_mentions = trainer.train()
        path = self.path + self._config['model']['config_path'] + (model_name if model_name is not None else "")
        self.model.saveModel(path)
        print 'Done!'

    def evaluate(self, config=None):
        if config is None:
            config = self._config["evaluation"]
        evaluation = Evaluation(self.iterators[config['iterator']],
                                self.model,
                                self.candidator,
                                self.stats[config['stats']],
                                sampling=config['sampling'] if 'sampling' in config else None,
                                log_path=self.path + "/evaluation.txt",
                                db=self.db,
                                trained_mentions=self.trained_mentions)
        evaluation.evaluate()

if __name__ == "__main__":
    experiment = Experiment("/experiments/wikilinks_deep_pointwise_no_attn/experiment.conf")
    for x in xrange(1):
        trained_mentions = experiment.train(model_name='.'+str(x))
        experiment.evaluate()
