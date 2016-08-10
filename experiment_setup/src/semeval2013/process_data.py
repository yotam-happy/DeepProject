from DbWrapper import *
import urllib2
from Word2vecLoader import *
import xml.etree.ElementTree as ET
from RNNModel import *
from KnockoutModel import *
from multiprocessing.pool import ThreadPool
from nltk.stem.wordnet import WordNetLemmatizer
from DbWrapper import *

_bbl_key = 'c2f19e32-3b8b-4f84-8e18-b68ad0b2a47f'
query_count = 1

def getSynsetsForWord(word, retries = 3):
    for i in xrange(retries):
        try:
            global query_count
            query_count += 1
            req = 'https://babelnet.io/v3/getSynsetIds?word={}&langs={}&key={}&source=WIKIWN,WIKI'
            json_str = urllib2.urlopen(req.format(word, 'EN', _bbl_key)).read()
            return json.loads(json_str)
        except:
            print "couldn't get synsets for ", word, " try ", i+1
            if i == retries:
                print "giving up on ", word
    return None

def getSynsetInfo(synsetId, retries = 3):
    for i in xrange(retries):
        try:
            global query_count
            query_count += 1
            req = 'https://babelnet.io/v3/getSynset?id={}&key={}'
            json_str = urllib2.urlopen(req.format(synsetId, _bbl_key)).read()
            return json.loads(json_str)
        except:
            print "couldn't get info for synset ", synsetId, " try ", i + 1
            if i == retries:
                print "giving up on ", synsetId
    return None

def getCandidateTitles(word):
    synsetIds = getSynsetsForWord(word)
    if synsetIds is None or len(synsetIds) == 0:
        return []
    pool = ThreadPool(processes=len(synsetIds))
    # create threads
    async = []
    for synset in synsetIds:
        id = synset['id']
        async.append(pool.apply_async(getSynsetInfo, (id,)))

    # read results
    wikiTitles = []
    for async_result in async:
        info = async_result.get()
        if info is None:
            continue
        for sense in info['senses']:
            if sense['language'] != "EN":
                continue
            if sense['source'] == 'WIKI':
                wikiTitles.append(sense['lemma'].lower())

    return wikiTitles

def semevalIterateAll(root):
    for text in root:
        for sentence in text:
            for word in sentence:
                wordType = word.tag
                wordId = word.attrib['id'] if 'id' in word.attrib else None
                yield (word.text, wordType, wordId)

def loadKey(key_f):
    key = dict()
    with open(key_f, "r") as f:
        for l in f.readlines():
            a = l.split()
            key[a[1]] = a[2]
    return key

def semevalIterateSentences(root, goldstandard_f = None):
    for text in root:
        for sentence in text:
            l = []
            for word in sentence:
                wordType = word.tag
                wordId = word.attrib['id'] if 'id' in word.attrib else None
                l.append((word.text, wordType, wordId))
            yield l


def toVec(root, key_f = None):
    '''
    Converts dataset to wikilinks format
    '''
    key = loadKey(key_f) if key_f is not None else None
    for sentence in semevalIterateSentences(root):
        for i in xrange(len(sentence)):
            word, wordType, wordId = sentence[i]
            if wordType == 'instance':
                wikilink = {}
                wikilink['word'] = word
                wikilink['token_id'] = wordId
                if key is not None:
                    if wordId not in key:
                        continue
                    wikilink['wikiId'] = key[wordId]
                if i != 0:
                    wikilink['left_context'] = [w for (w,t,id) in sentence[:i]]
                if i < len(sentence) - 1:
                    wikilink['right_context'] = [w for (w,t,id) in sentence[i+1:]]
                yield wikilink

def convWord(w):
        try:
            w = w.decode('unicode-escape')
        except:
            print "couldn't decode ", w
        return w.lower()


if __name__ == "__main__":
    key_path = "../data/semeval-2013-task12-test-data/keys/gold/wikipedia/wikipedia.en.key"
    tree = ET.parse('../data/semeval-2013-task12-test-data/data/multilingual-all-words.en.xml')
    root = tree.getroot()

    # get all words to disambiguate
    words = set()
    key = loadKey(key_path)
    lmtzr = WordNetLemmatizer()
    for (word, wordType, wordId) in semevalIterateAll(root):
        if (wordType == 'instance' and wordId in key):
            words.add(convWord(word))
    print words

    # get all candidates from babelnet
    global query_count
    query_count = 0
    wordSenses = dict()
    for i, word in enumerate(words):
        wordSenses[word] = [convWord(c) for c in getCandidateTitles(word)]
        lemma = lmtzr.lemmatize(word)
        if (lemma != word):
            wordSenses[word] += [convWord(c) for c in getCandidateTitles(lmtzr.lemmatize(word))]
        print word, " (lemma: ", lmtzr.lemmatize(word), ") : ", wordSenses[word]
        if i % 10 == 0:
            print "done ", i, " (with ", query_count, " queries)"

    f = open("semeval_senses",'wb')
    pickle.dump(wordSenses, f)
    f.close()

    f = open('semeval_senses', 'rb')
    wordSenses = pickle.load(f)
    f.close()

    wikiDB = WikipediaDbWrapper(user='root', password='rockon123', database='wikiprep-esa-en20151002')
    count = 0
    no_word = 0
    no_sense = 0
    not_in_db = 0
    for wlink in toVec(root,key_f=key_path):
        count += 1
        if convWord(wlink['word']) not in wordSenses:
            no_word += 1
            print "no word: ", wlink['word']
        if convWord(wlink['wikiId']) not in wordSenses[convWord(wlink['word'])]:
            no_sense += 1
            print "no sense: ", convWord(wlink['wikiId']), " for word ", convWord(wlink['word']) \
                , " (senses: ", wordSenses[convWord(wlink['word'])] ,")"
        if wikiDB.resolvePage(wlink['wikiId']) is None:
            not_in_db += 1
            print wlink['wikiId'], " is not in our DB (strange...)"
    print "count ", count, " no word: ", no_word, " no_sense: ", no_sense, " not in db: ", not_in_db

    ## SOMETHING ELSE: try to predict
    path = "/home/yotam/pythonWorkspace/deepProject"
    print "Loading iterators+stats..."
    if(not os.path.isdir(path)):
        path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

    train_stats = WikilinksStatistics(None, load_from_file_path=path + "/data/wikipedia-intralinks.stats")
    print "Done!"

    print 'Loading embeddings...'
    w2v = Word2vecLoader(wordsFilePath=path + "/data/word2vec/dim300vecs",
                         conceptsFilePath=path + "/data/word2vec/dim300context_vecs")
    wD = train_stats.mentionLinks
    cD = train_stats.conceptCounts
    w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
    print 'wordEmbedding dict size: ',len(w2v.wordEmbeddings)
    print 'conceptEmbeddings dict size: ',len(w2v.conceptEmbeddings)
    print 'Done!'

    print "loading model"
    pairwise_model = RNNPairwiseModel(w2v)
    pairwise_model.loadModel(path + "/models/wiki-intralinks.0.out")
    knockout_model = KnockoutModel(pairwise_model,train_stats)
    print 'Done!'

    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    tree = ET.parse(path + '/data/semeval-2013-task12-test-data/data/multilingual-all-words.en.xml')
    root = tree.getroot()
    for wlink in toVec(root):
        a = knockout_model.predict(wlink)
        print wlink
        cc = train_stats.getCandidatesForMention(wlink['word'])
        candidates = []
        if cc is not None:
            print cc
            candidates = [wikiDB.getConceptTitle(int(c)) for c in cc.keys()]
        print "candidates: ", candidates
        print "predicted: ", 'None' if a is None else wikiDB.getConceptTitle(int(a))
