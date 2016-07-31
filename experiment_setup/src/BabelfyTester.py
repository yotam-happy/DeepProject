import gzip
import urllib
import urllib2
from StringIO import StringIO
import json


class BabelfyTester:

    def __init__(self, db):
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
        input = (' '.join(wlink['right_context'])) + ' ' + wlink['word'] + ' ' +  ' '.join(wlink['left_context'])
        char_index_start = len(' '.join(wlink['right_context'])) + 1 # the char fragment index of the word
        char_index_end = len((' '.join(wlink['right_context'])) + ' ' + wlink['word'] ) - 1 # the char fragment index of the word

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
            match_sensitive_flag = 0 # flag is up if we have a full match
            for result in data:

                # retrieving char fragment
                charFragment = result.get('charFragment')
                cfStart = charFragment.get('start')
                cfEnd = charFragment.get('end')

                # for every charfragment with word find the wiki lemma
                # case 1 - we have a full fragment match and nned to choose among different senses
                if cfStart == char_index_start and cfEnd == char_index_end and result.get('score') >= default_score:
                    match_sensitive_flag = 1
                    default_score = result.get('score')
                    synsetId = result.get('babelSynsetID')
                # case 2 - we have partial fragment match (expressions start with same char) so we test this case (for example 'Jaguar' and 'Jaguar car')
                elif cfStart == char_index_start and result.get('score') >= default_score and match_sensitive_flag == 0:
                    default_score = result.get('score')
                    synsetId = result.get('babelSynsetID')

            # retrive babelnet lemma
            if synsetId is None:
                return None
            lemma_cand = (self.retriveLemma(synsetId)).lower()
            if lemma_cand is not None:
                print " predicted: ", lemma_cand, ' actual: ', wlink['wikiurl']

            return self.db.resolvePage(lemma_cand) if lemma_cand is not None else None

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




