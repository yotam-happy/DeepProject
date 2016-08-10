import gzip
import urllib
import urllib2
from StringIO import StringIO
import json
import re
from WikilinksProcessing import wlink_writer

class BabelfyTester:

    def __init__(self, db, saveGoodPath=None):
        '''
        :param db:
        :param saveGoodPath:    if this is not null then we save the wlinks where babelfy returned a response that fits our needs
                                into the given path (should be a directory name)
        '''
        self.db = db
        self.service_url_bf = 'https://babelfy.io/v1/disambiguate'
        self.service_url_bn = 'https://babelnet.io/v3/getSynset'
        self.key = 'dfb53f62-b530-46ce-8245-60fa13ecb763'
        self.params_bf = {
            'text': '',
            'lang': 'EN',
            'key': self.key,
            'cands': 'TOP',
        }
        self.params_bn = {
            'id' : ' ',
            'key'  : self.key,
        }

        self._tried = 0
        self._correct = 0
        self._writer = wlink_writer(saveGoodPath) if saveGoodPath is not None else None

    def finalizeWriter(self):
        self._writer.finalize()

    def predict(self, wlink):
        """
        @:param wlink - input wikilink
        @:returns wikiid
        This function takes a wikilink, finds the word field string babelsynt with Babelfy, and returns
        the WIKI source lemma from Babelnet. In case of multiple candidatesm were several senses start from
        the same char_indx as the wikilink word, the function takes the sense with the highest score field according
        to Babelfy.
        """

        # create input text and send to babelfy
        default_score = 0
        lemma_cand = None
        input = wlink['left_context_text'] + ' ' + wlink['word'] + ' ' + wlink['right_context_text']
        char_indx = len(wlink['left_context_text']) + 1 # the char fragment index of the word
        char_indx_end = len(wlink['left_context_text']) + len(wlink['word']) # the char fragment index of the word

        # print some of the context

        self.params_bf['text'] = str(input.encode('utf-8'))
        url = self.service_url_bf + '?' + urllib.urlencode(self.params_bf)
        request = urllib2.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib2.urlopen(request)

        if response.info().get('Content-Encoding') == 'gzip':
            buf = StringIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read())
            synsetId = None
            for result in data:

                # retrieving char fragment
                charFragment = result.get('charFragment')
                cfStart = charFragment.get('start')
                cfEnd= charFragment.get('end')

                # for every charfragment with word find the wiki lemma
                if cfStart == char_indx and cfEnd == char_indx_end and result.get('score') >= default_score:
                    default_score = result.get('score')
                    synsetId = result.get('babelSynsetID')

            # retrive babelnet lemma
            if synsetId is None:
                return None
            lemma_cand = self.retriveLemma(synsetId)
            if lemma_cand is not None:
                actual = wlink['wikiurl']
                if actual.rfind('/') > -1:
                    actual = actual[actual.rfind('/')+1:]
                if actual.find('#') > -1:
                    actual = actual[:actual.find('#')]
                regex = re.compile('[^\w]')
                actual = regex.sub('', actual).lower()
                pred = regex.sub('', lemma_cand).lower()

                print " prdicted: ", pred, ' actual: ', actual, ' word: ', wlink['word']

                self._tried += 1
                if actual == pred:
                    self._correct += 1
                if self._tried % 10 == 0:
                    print "Babelfy test: ", float(self._correct) / self._tried

            p = self.db.resolvePage(lemma_cand) if lemma_cand is not None else None
            if p is not None:
                self._writer.save(wlink)
            return p

    def retriveLemma(self, bn_synt):
        # retriving the lemma ofa given babelnetsynt with babelnet API

        self.params_bn['id'] = bn_synt
        url_bn = self.service_url_bn + '?' + urllib.urlencode(self.params_bn)
        request_bn = urllib2.Request(url_bn)
        request_bn.add_header('Accept-encoding', 'gzip')
        response_bn = urllib2.urlopen(request_bn)

        if response_bn.info().get('Content-Encoding') == 'gzip':
            buf = StringIO( response_bn.read())
            f = gzip.GzipFile(fileobj=buf)
            data_bn = json.loads(f.read())

            ## retrieving BabelSense data
            senses = data_bn['senses']
            for result in senses:
                source = result.get('source')
                lemma = result.get('lemma')
                if(str(source.encode('utf-8')) == 'WIKI'):
                    return (lemma.encode('utf-8'))




