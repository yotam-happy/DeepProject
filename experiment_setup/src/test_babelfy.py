"""
using Babelfy API for testing its performance on our test set
"""

import requests as r
import urllib2
import urllib
import json
import gzip
import ujson as json
from StringIO import StringIO

from BabelfyTester import BabelfyTester
from WikilinksIterator import *
from WikilinksStatistics import *
from Word2vecLoader import *

## ------------ code for testing the BabelfyTester ------------

print "Loading iterators+stats..."
path = "C:\\repo\\DeepProject"
if(not os.path.isdir(path)):
    path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

train_stats = WikilinksStatistics(None, load_from_file_path=path+"\\data\\wikilinks\\train_stats")
iter_train = WikilinksNewIterator(path+"\\data\\wikilinks\\small_train",
                                  mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
iter_eval = WikilinksNewIterator(path+"\\data\\wikilinks\\small_evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
wl_itr = iter_eval.wikilinks()
bfy = BabelfyTester(0)

# run the next 2 lines to check different wikilinks
wlink = wl_itr.next()
bfy.predict(wlink) # TODO: delete the prints and take the lemma corresponding wikiID from table

## ------------ testing babelfy ------------
dissamb_service_url = 'https://babelfy.io/v1/disambiguate'
##
text = 'I went fishing for some sea bass.'
lang = 'EN'
key = 'dfb53f62-b530-46ce-8245-60fa13ecb763'
cands = 'TOP'
params = {
    'text': text,
    'lang': lang,
    'key': key,
    'cands': cands,
}

url = dissamb_service_url+ '?' + urllib.urlencode(params)
request = urllib2.Request(url)
request.add_header('Accept-encoding', 'gzip')
response = urllib2.urlopen(request)

#
if response.info().get('Content-Encoding') == 'gzip':
    buf = StringIO(response.read())
    f = gzip.GzipFile(fileobj=buf)
    data = json.loads(f.read())

# retrieving data
for result in data:
    # retrieving token fragment
    tokenFragment = result.get('tokenFragment')
    tfStart = tokenFragment.get('start')
    tfEnd = tokenFragment.get('end')
    print str(tfStart) + "\t" + str(tfEnd)

    # retrieving char fragment
    charFragment = result.get('charFragment')
    cfStart = charFragment.get('start')
    cfEnd = charFragment.get('end')
    print str(cfStart) + "\t" + str(cfEnd)

    # retrieving BabelSynset ID
    synsetId = result.get('babelSynsetID')
    print synsetId

    # displaying the url
    DBurl = result.get('DBpediaURL')
    print DBurl

## ------------ testing babelnet ------------
"""
find the wiki id
"""

synset_service_url = 'https://babelnet.io/v3/getSynset'

# manga id and word
wikilink_freebaseid = 'Some(9202a8c04000641f800000000384c5ba)'
word = 'manga'
id = 'bn:01163691n'

params = {
	'id' : id,
	'key'  : key
}

url = synset_service_url + '?' + urllib.urlencode(params)
request = urllib2.Request(url)
request.add_header('Accept-encoding', 'gzip')
response = urllib2.urlopen(request)

if response.info().get('Content-Encoding') == 'gzip':
    buf = StringIO( response.read())
    f = gzip.GzipFile(fileobj=buf)
    data = json.loads(f.read())

    ## retrieving BabelSense data
    senses = data['senses']
    for result in senses:
        source = result.get('source')
        lemma = result.get('lemma')
        language = result.get('language')
        if(str(source.encode('utf-8')) == 'WIKI'):
            print str(lemma.encode('utf-8')) +"\t" + str(source.encode('utf-8'))

    print '\n'
    ## retrieving BabelGloss data
    glosses = data['glosses']
    for result in glosses:
    	gloss = result.get('gloss')
    	language = result.get('language')
    	print language.encode('utf-8') + "\t" + str(gloss.encode('utf-8'))

    print '\n'
    ## retrieving BabelImage data
    images = data['images']
    for result in images:
    	url = result.get('url')
    	language = result.get('language')
    	name = result.get('name')
    	print language.encode('utf-8') +"\t"+ str(name.encode('utf-8')) +"\t"+ str(url.encode('utf-8'))


