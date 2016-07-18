import os
from DbWrapper import *
import urllib2
from urllib2 import HTTPError
from Word2vecLoader import *
import ujson as json
import mysql
import xml.etree.ElementTree as ET
from RNNModel import *
from KnockoutModel import *
import mysql.connector
from multiprocessing.pool import ThreadPool

_bbl_key = 'c2f19e32-3b8b-4f84-8e18-b68ad0b2a47f'
query_count = 1

def getSynsetsForWord(word):
    try:
        global query_count
        query_count += 1
        req = 'https://babelnet.io/v3/getSynsetIds?word={}&langs={}&key={}&source=WIKIWN,WIKI'
        json_str = urllib2.urlopen(req.format(word, 'EN', _bbl_key)).read()
        return json.loads(json_str)
    except:
        print "couldn't get synsets for ", word
        return None

def getSynsetInfo(synsetId):
    try:
        global query_count
        query_count += 1
        req = 'https://babelnet.io/v3/getSynset?id={}&key={}'
        json_str = urllib2.urlopen(req.format(synsetId, _bbl_key)).read()
        return json.loads(json_str)
    except:
        print "couldn't get info for synset ", synsetId
        return None

def getCandidateTitles(word):
    synsetIds = getSynsetsForWord(word)
    if synsetIds is None:
        return []
    wikiTitles = []
    for synset in synsetIds:
        id = synset['id']
        info = getSynsetInfo(id)
        if info is None:
            continue
        for sense in info['senses']:
            if sense['language'] != "EN":
                continue
            if sense['source'] == 'WIKI':
                wikiTitles.append(sense['lemma'])

    return wikiTitles

def getCandidateTitles(word):
    synsetIds = getSynsetsForWord(word)
    if synsetIds is None:
        return []
    wikiTitles = []
    for synset in synsetIds:
        id = synset['id']
        info = getSynsetInfo(id)
        if info is None:
            continue
        for sense in info['senses']:
            if sense['language'] != "EN":
                continue
            if sense['source'] == 'WIKI':
                wikiTitles.append(sense['lemma'])

    return wikiTitles

def getCandidateTitlesMultiThread(word):
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
                wikiTitles.append(sense['lemma'])

    return wikiTitles

def semevalIterateAll(root):
    for text in root:
        for sentence in text:
            for word in sentence:
                wordType = word.tag
                wordId = word.attrib['id'] if 'id' in word else None
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

if __name__ == "__main__":
    key_path = "C:\\repo\\DeepProject\\data\\semeval-2013-task12-test-data\\keys\\gold\\wikipedia\\wikipedia.en.key"
    tree = ET.parse('C:\\repo\\DeepProject\\data\\semeval-2013-task12-test-data\\data\\multilingual-all-words.en.xml')
    root = tree.getroot()

    # get all words to disambiguate
    words = []
    key = loadKey(key_path)
    for (word, wordType, wordId) in semevalIterateAll(root):
        if (wordType == 'instance' and wordId in key):
            words.append(word)


    # get all candidates from babelnet
    global query_count
    query_count = 0
    wordSenses = dict()
    for i, word in enumerate(words):
        wordSenses[word] = getCandidateTitlesMultiThread(word)
        print word, " ", wordSenses[word]
        if i % 10 == 0:
            print "done ", i, " (with ", query_count, " queries)"

    f = open("semeval_senses",'wb')
    pickle.dump(wordSenses, f)
    f.close()

    count = 0
    got_it = 0
    no_word = 0
    no_sense = 0
    for wlink in toVec(root,key_f=key_path):
        count += 1
        if wlink['word'] not in wordSenses:
            no_word += 1
            print "no word: ", wlink['word']
        if wlink['wikiId'] not in wordSenses[wlink['word']]:
            no_sense += 1
            print "no sense: ", wlink['wikiId'], " for word ", wlink['word'], " (senses: ", wordSenses[wlink['word']] ,")"
    print "count ", count, "got ", got_it, " no word: ", no_word, " no_sense: ", no_sense

    getSynsetsForWord('intention')
    getSynsetInfo('bn:00076732n')
    getCandidateTitles('text')


    ## SOMETHING ELSE: try to predict
    path = "C:\\repo\\DeepProject"
    print "Loading iterators+stats..."
    if(not os.path.isdir(path)):
        path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

    train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
    print "Done!"

    print 'Loading embeddings...'
    w2v = Word2vecLoader(wordsFilePath=path+"\\data\\word2vec\\dim300vecs",
                         conceptsFilePath=path+"\\data\\word2vec\\dim300context_vecs")
    wD = train_stats.mentionLinks
    cD = train_stats.conceptCounts
    w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
    print 'wordEmbedding dict size: ',len(w2v.wordEmbeddings)
    print 'conceptEmbeddings dict size: ',len(w2v.conceptEmbeddings)
    print 'Done!'

    print "loading model"
    pairwise_model = RNNPairwiseModel(w2v)
    pairwise_model.loadModel(path + "\\models\\rnn.relu6")
    knockout_model = KnockoutModel(pairwise_model,train_stats)
    print 'Done!'

    wikiDB = WikipediaDbWrapper(user='root', password='rockon123', database='wikiprep-esa-en20151002')
    tree = ET.parse('C:\\repo\\DeepProject\\data\\semeval-2013-task12-test-data\\data\\multilingual-all-words.en.xml')
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
