"""
Mannually execute each class dependency to work with the project without compiling
Best and easiest way to debug code in python. You can actully view to whole workspace

ATTENTION: Change the path variable in the end of the file
ATTENTION: If you de modificaitons in the classes please keep the copyies here updated! (There must be a better way but I am lazy)

NOTEs: use collapse all shortcut CTRL + SHIFT + NumPad (-)  to navigate and excecute easily the code.
also use the ALT+SHIFT+E to execute single lines or whole code fragments
I also recommend on Pycharm cell mode plugin for easier execution of code fragments
(Noam)
"""

## The cell seperator
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import ujson as json
import nltk
import unicodedata
import random
import pickle
from scipy import spatial
import numpy as np
import os
from keras.callbacks import EarlyStopping

##
class KnockoutModel:
    """
    This model takes a pairwise model that can train/predict on pairs of candidates for a wikilink
    and uses it to train/predict on a list candidates using a knockout method.
    The candidates are taken from a stats object
    """

    def __init__(self, pairwise_model, stats):
        """
        :param pairwise_model:  The pairwise model used to do prediction/training on a triplet
                                (wikilink,candidate1,candidate2)
        :param stats:           A statistics object used to get list of candidates
        """
        self._stats = stats
        self._pairwise_model = pairwise_model

    def predict(self, wikilink):
        candidates = self._stats.getCandidatesForMention(wikilink["word"])
        if candidates is None:
            return None

        # do a knockout
        l = candidates.keys()
        while len(l) > 1:
            # create a list of surviving candidates by comparing couples
            next_l = []

            for i in range(0, len(l) - 1, 2):
                a = self._pairwise_model.predict(wikilink, l[i], l[i+1])
                if a is not None:
                    next_l.append(a)

            if len(l) % 2 == 1:
                next_l.append(l[-1])
            l = next_l

        if len(l) == 0:
            return None
        return l[0]

    def train(self, wikilink):
        # TODO: change from default predict definitions to not-complete knockout
        candidates = self._stats.getCandidatesForMention(wikilink["word"])
        if candidates is None:
            return None

        # do a knockout
        l = candidates.keys()
        while len(l) > 1:
            # create a list of surviving candidates by comparing couples
            next_l = []
            for i in range(0, len(l) - 1, 1):
                a = self._pairwise_model.train(wikilink, wikilink["wikiId"], l[i], correct= wikilink["wikiId"])
                if a is not None:
                    next_l.append(a)

            if len(l) % 2 == 1:
                next_l.append(l[-1])
            l = next_l

        if len(l) == 0:
            return None
        return l[0]

class WikilinksNewIterator:

    # the new iterator does not support using a zip file.
    def __init__(self, path, limit_files = 0, mention_filter=None):
        self._path = path
        self._limit_files = limit_files
        self._mention_filter = mention_filter

    def _wikilink_files(self):
        for file in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, file)):
                continue
            print "opening ", file
            yield open(os.path.join(self._path, file), 'r')

    def wikilinks(self):
        for c, f in enumerate(self._wikilink_files()):
            lines = f.readlines()
            for line in lines:
                if len(line) > 0:
                    wlink = json.loads(line)

                    # filter
                    if (not 'word' in wlink) or (not 'wikiId' in wlink):
                        continue
                    if not ('right_context' in wlink or 'left_context' in wlink):
                        continue
                    if self._mention_filter is not None and wlink['word'] not in self._mention_filter:
                        continue

                    # preprocess context
                    if 'right_context' in wlink:
                        r_context = unicodedata.normalize('NFKD', wlink['right_context']).encode('ascii','ignore').lower()
                        wlink['right_context'] = nltk.word_tokenize(r_context)
                    if 'left_context' in wlink:
                        l_context = unicodedata.normalize('NFKD', wlink['left_context']).encode('ascii','ignore').lower()
                        wlink['left_context'] = nltk.word_tokenize(l_context)

                    # return
                    yield wlink

            f.close()
            if self._limit_files > 0 and c >= self._limit_files:
                break

class WikilinksRewrite:
    """
        This class takes an iterator of the dataset and creates a new dataset where
        each wikilink is its own json contained in one line.
        This is opposed to the old style we had where the entire file was a single
        json and each wikilink was in multiple lines.

        wikilinks_iter -    an iterator of the dataset, supposed to be of the old
                            style so this class converts to new style
    """

    def __init__(self, wikilinks_iter, dest_dir, json_per_file=400000):
        self._iter = wikilinks_iter
        self._dest_dir = dest_dir
        self._n = 0
        self._json_per_file = json_per_file

    def _open_file(self, n):
        return open(os.path.join(self._dest_dir, 'wikilinks_{}.json'.format(n)), mode='w')

    def work(self):
        l = []
        for wikilink in self._iter.wikilinks():
            l.append(json.dumps(wikilink))
            if len(l) >= self._json_per_file:
                # write list to file
                f = self._open_file(self._n)
                for s in l:
                    f.write(s + '\n')
                f.close()
                self._n += 1
                l = []

class ShuffleFiles:
    """
    This class takes a source directory which is assumed to contain some text file.
    It then writes the contents of these files into dest_dir, into a similar number
    of files but with the lines: roughly equally devided between the files and randomly
    shuffled both between the files and inside the files.

    the process is a two step process and one must call work1() and then work2() to do
    the job
    """
    def __init__(self, src_dir, dest_dir):
        self._src_dir = src_dir
        self._dest_dir = dest_dir

    def _open_for_write(self, dir, n):
        return open(os.path.join(dir, 'wikilinks_{}.json'.format(n)), mode='w')

    # step 1 of randomizing
    def work1(self):
        # open files for write
        dest_files = [self._open_for_write(self._dest_dir, n) for n in xrange(len(os.listdir(self._src_dir)))]
        print "first phase..."

        for fname in os.listdir(self._src_dir):
            in_f = open(os.path.join(self._src_dir, fname), 'r')
            dest_files_temp = [[] for n in xrange(len(dest_files))]
            for line in in_f:
                dest_files_temp[random.randrange(len(dest_files))].append(line)
            in_f.close()

            for f, l in zip(dest_files, dest_files_temp):
                    f.writelines(l)
            print "done ", fname

        for f in dest_files:
            f.close()

    # step 2 of randomizing
    def work2(self):
        print "second phase..."
        for fname in os.listdir(self._dest_dir):
            print "opening file: " + fname
            f = open(os.path.join(self._dest_dir, fname), 'r')
            l = f.readlines()
            f.close()

            random.shuffle(l)

            f = open(os.path.join(self._dest_dir, fname), 'w')
            f.writelines(l)
            f.close()

class Word2vecLoader:
    """
    Words a word2vec model. Loads both word vectors and concept (context) vectors. These must
    match in vector sizes.
    Notice the files are not loaded when the object is created and one must explicitly
    call loadEmbeddings() to do so
    """

    def __init__(self, wordsFilePath="vecs", conceptsFilePath="context"):
        self._wordsFilePath = wordsFilePath
        self._conceptsFilePath = conceptsFilePath

        self.wordEmbeddings = dict()
        self.conceptEmbeddings = dict()

        # make sure embedding sizes match
        with open(wordsFilePath) as f:
            _, sz = f.readline().split()
            self.embeddingSize = int(sz)

        with open(conceptsFilePath) as f:
            _, embeddingSz = f.readline().split()
            if int(embeddingSz) != self.embeddingSize:
                raise Exception("Embedding sizes don't match")

    def _loadEmbedding(self, path, filterSet):
        embedding = dict()
        with open(path) as f:
            f.readline() # skip embedding size def
            for line in iter(f):
                s = line.split()
                if filterSet is None or s[0] in filterSet:
                    embedding[s[0].lower()] = np.array([float(x) for x in s[1:]])
        return embedding

    def meanOfWordList(self, l):
        sum = np.zeros(self.embeddingSize)
        k = 0
        for w in l:
            if w in self.wordEmbeddings:
                sum += self.wordEmbeddings[w]
                k += 1
        return sum / k

    def distance(self, v1, v2):
        return spatial.distance.cosine(v1,v2)

    def loadEmbeddings(self, wordDict=None, conceptDict=None):
        """
        Loads both word and context embeddings.

        :param wordDict: If specified, only words appearing in word dict will be kept in memory
        :param conceptDict: If specified, only concepts appearing in concept dict will be kept in memory
        """
        parent_path = os.path.abspath(os.path.join(self._wordsFilePath, os.pardir))
        if(os.path.isfile(parent_path+'\\w2v_filt.txt')):
            file_w2v = open(parent_path+'\\w2v_filt.txt','rb')
            self.wordEmbeddings = pickle.load(file_w2v)
            self.conceptEmbeddings = pickle.load(file_w2v)
            file_w2v.close()
        else:
            self.wordEmbeddings = self._loadEmbedding(self._wordsFilePath, wordDict)
            self.conceptEmbeddings = self._loadEmbedding(self._conceptsFilePath, conceptDict)
            file_w2v = open("..\\..\\data\\word2vec\\w2v_filt.txt",'ab+')
            pickle.dump(self.wordEmbeddings, file_w2v)
            pickle.dump(self.conceptEmbeddings, file_w2v)
            file_w2v.close()

class WikilinksStatistics:
    """
    This class can calculate a number of statistics regarding the
    wikilink dataset.

    To calculate the statistics one needs to call calcStatistics() method.
    The class will then populate the following member variables:

    mentionCounts       dictionary of mention=count. Where mention is a surface term to be disambiguated
                        and count is how many times it was seen n the dataset
    conceptCounts       dictionary of concept=count. Where a concept is a wikipedia id (a sense), and count
                        is how many times it was seen (how many mentions refered to it)
    contextDictionary   dictionary of all words that appeared inside some context (and how many times)
    mentionLinks        holds for each mention a dictionary of conceptsIds it was reffering to and how many
                        times each. (So its a dictionary of dictionaries)
    """

    def __init__(self, wikilinks_iter, load_from_file_path=None):
        """
        Note the statistics are not calculated during init. (unless loaded from file)
        so must explicitly call calcStatistics()
        :param wikilinks_iter:      Iterator to a dataset
        :param load_from_file_path: If given then the statistics are loaded from this file
        """
        self._wikilinks_iter = wikilinks_iter
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.contextDictionary = dict()
        if load_from_file_path is not None:
            self.loadFromFile(load_from_file_path)


    def saveToFile(self, path):
        """ saves statistics to a file """
        f = open(path, mode='w')
        f.write(json.dumps(self.mentionCounts)+'\n')
        f.write(json.dumps(self.mentionLinks)+'\n')
        f.write(json.dumps(self.conceptCounts)+'\n')
        f.write(json.dumps(self.contextDictionary))
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.conceptCounts = json.loads(l[2])
        self.contextDictionary = json.loads(l[3])
        f.close()

    def calcStatistics(self):
        """
        calculates statistics and populates the class members. This should be called explicitly
        as it might take some time to complete. It is better to call this method once and save
        the results to a file if the dataset is not expected to change
        """
        print "getting statistics"
        for wlink in self._wikilinks_iter.wikilinks():
            if not wlink['word'] in self.mentionLinks:
                self.mentionLinks[wlink['word']] = dict()
            self.mentionLinks[wlink['word']][wlink['wikiId']] = self.mentionLinks[wlink['word']].get(wlink['wikiId'], 0) + 1
            self.mentionCounts[wlink['word']] = self.mentionCounts.get(wlink['word'], 0) + 1
            self.conceptCounts[wlink['wikiId']] = self.conceptCounts.get(wlink['wikiId'], 0) + 1

            if 'right_context' in wlink:
                for w in wlink['right_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1
            if 'left_context' in wlink:
                for w in wlink['left_context']:
                    self.contextDictionary[w] = self.contextDictionary.get(w, 0) + 1

    def getCandidatesForMention(self, mention):
        """
        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        if mention not in self.mentionLinks:
            return None
        return self.mentionLinks[mention]

    def getGoodMentionsToDisambiguate(self, f=5):
        """
        Returns a set of mentions that are deemed "good"
        These are mentions where the second most common sense appears at least 10 times
        :param f:
        :return:
        """

        # generates a list of mentions, sorted by the second most common sense per
        # mention
        k, v = self.mentionLinks.items()[0]
        l = [(k, self._sortedList(v)) for k,v in self.mentionLinks.items()]

        # take those mentions where the second most common term appears more then f times
        s = set()
        for mention in l:
            if len(mention[1]) > 1 and mention[1][1][1] >= f:
                s.add(mention[0])
        return s

    def _sortedList(self, l):
        l = [(k,v) for k,v in l.items()]
        l.sort(key=lambda (k,v):-v)
        return l

    def printSomeStats(self):
        """
        Pretty printing of some of the statistics in this object
        """

        print "distinct terms: ", len(self.mentionCounts)
        print "distinct concepts: ", len(self.conceptCounts)
        print "distinct context words: ", len(self.contextDictionary)

        k, v = stats.mentionLinks.items()[0]
        wordsSorted = [(k, self._sortedList(v), sum(v.values())) for k,v in stats.mentionLinks.items()]
        wordsSorted.sort(key=lambda (k, v, d): v[1][1] if len(v) > 1 else 0)

        print("some ambiguous terms:")
        for w in wordsSorted[-10:]:
            print w

class Evaluation:
    """
    This class evaluates a given model on the dataset given by test_iter.
    """

    def __init__(self, test_iter, model):
        """
        :param test_iter:   an iterator to the test or evaluation set
        :param model:       a model to evaluate
        """
        self._iter = test_iter
        self._model = model

        self.n_samples = 0
        self.correct = 0
        self.no_prediction = 0

    def evaluate(self, mode="predict"):
        """
        Do the work - runs over the given test/evaluation set and compares
        the predictions of the model to the actual sense.

        Populates the members:
        self.n_samples:     number of samples we tested on
        self.correct:       number of correct predictions
        self.no_prediction: number of samples the model did not return any prediction for

        :return:
        """
        self.n_samples = 0
        self.correct = 0
        self.no_prediction = 0

        for wikilink in self._iter.wikilinks():
            if 'wikiId' not in wikilink:
                continue
            actual = wikilink['wikiId']

            if(mode == 'predict'):
                prediction = self._model.predict(wikilink)

                self.n_samples += 1
                if prediction is None:
                    self.no_prediction += 1
                elif prediction == actual:
                    self.correct += 1

                if(self.n_samples % 1000 == 0):
                    print 'sampels=', self.n_samples ,'; %correct=', float(self.correct) / (self.n_samples - self.no_prediction)

            if(mode == 'train'):
                # TODO: define stopping criteria for training
                self._model.train(wikilink)

        self.printEvaluation(mode)

    def printEvaluation(self,mode):
        """
        Pretty print results of evaluation
        """
        if mode == 'train':
            print "TODO: print evaluation on training set..."

        if mode == 'predict':
            print "samples: ", self.n_samples, "; correct: ", self.correct, " no-train: ", self.no_prediction
            print "%correct from total: ", float(self.correct) / self.n_samples
            print "%correct where prediction was attempted: ", float(self.correct) / (self.n_samples - self.no_prediction)

def getVanillaNNPairwiseModel(train_stats,w2v):
    vanilla_nn_model = VanillaNNPairwiseModel(w2v)
    return KnockoutModel(vanilla_nn_model,train_stats)

class VanillaNNPairwiseModel:
    """
    This model is a baseline NN with simple architecture. It has an prediction
    and train function and is part of the knockoutmodel system.
    the input of the model is assumed to be 3 embedding vectors with shape (300,) each
    for the mention, the candidate sense and the context embedding sense . the output should be a simple (2,)
    binary vector, specifying which candidate is more accurate
    """

    def __init__(self, w2v):
        self._w2v = w2v
        self._batchX = []
        self._batchY = []

        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons
        self._batch_size = 32
        # self._nb_epoch = 1e3 # FIXME
        self.model = Sequential()
        self.model.add(Dense( 64 ,input_dim = w2v.embeddingSize * 3 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense( 64 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2, init = 'uniform'))
        self.model.add(Activation('softmax'))

        # defining solver and compile
        sgd = SGD(lr=0.1, decay=1e-6,momentum=0.9)
        self.model.compile(loss='binary_crossentropy',optimizer='sgd')

    def train(self, wikilink, candidate1, candidate2, correct):
        """
        Takes a single example to train
        :param wikilink:    The wikilink to train on
        :param candidate1:  the first candidate
        :param candidate2:  the second candidate
        :param correct:     which of the two is correct (expected output)
        """

        if candidate1 not in self._w2v.conceptEmbeddings and candidate2 not in self._w2v.conceptEmbeddings:
            return None
        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        if candidate1 not in self._w2v.conceptEmbeddings:
            return candidate2
        if candidate2 not in self._w2v.conceptEmbeddings:
            return candidate1

        candidate1_vec = self._w2v.conceptEmbeddings[candidate1.lower()]
        candidate2_vec = self._w2v.conceptEmbeddings[candidate2.lower()]
        context_vec = self._w2v.meanOfWordList(wikilink['right_context'] + wikilink['left_context'])

        X = (np.asarray([context_vec, candidate1_vec, candidate2_vec])).flatten()
        Y = np.array([1,0] if candidate1 == correct else [0,1])
        self._trainXY(X,Y)

    def _trainXY(self,X,Y):
        self._batchX.append(X)
        self._batchY.append(Y)

        if len(self._batchX) >= self._batch_size:
            # pushes numeric data into batch vector
            batchX = np.array(self._batchX)
            batchY = np.array(self._batchY)

            # training on batch is specifically good for cases were data doesn't fit into memory
            print 'train batch. Size of batch x - ', batchX.shape
            self.model.train_on_batch(batchX, batchY)
            # print self.model.metrics_names

            self._batchX = []
            self._batchY = []

    def saveModel(self, fname):
        return

    def loadModel(self, fname):
        return

    def startTraining(self):
        return

    def finilizeTraining(self):
        return

    def predict(self, wikilink, candidate1, candidate2):
        return None

##
"""
here we test the VanillaNN structure
This is the main script
"""
print 'Starts model evluation\nStarts loading files...'
os.chdir("C:\Users\Noam\Documents\GitHub\DeepProject") # TODO: Yotam, you need to change this in order to work with this file
path = "C:\Users\Noam\Documents\GitHub\DeepProject"
iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\train")
train_stats = WikilinksStatistics(iter_train, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
print len(train_stats.getGoodMentionsToDisambiguate(f=10))
iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
w2v = Word2vecLoader(wordsFilePath=path+"\\data\\word2vec\\dim300vecs",
                     conceptsFilePath=path+"\\data\\word2vec\\dim300context_vecs")
wD = train_stats.mentionLinks
cD = train_stats.conceptCounts
print 'Load embeddings...'
w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)

## TRAIN DEBUGGING CELL
model = getVanillaNNPairwiseModel(train_stats,w2v)
ev = Evaluation(iter_eval, model)

ev.evaluate('train')
print 'Training...'

##
print 'Prediction...'
ev.evaluate('predict')

##