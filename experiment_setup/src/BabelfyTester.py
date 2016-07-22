import gzip
import urllib
import urllib2
from StringIO import StringIO
import json


class BabelfyTester:

    def __init__(self, wiki2id_table):
        self.wiki2id = wiki2id_table
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
        char_indx = len(' '.join(wlink['right_context'])) + 1 # the char fragment index of the word
        self.params_bf['text'] = input
        url = self.service_url_bf + '?' + urllib.urlencode(self.params_bf)
        request = urllib2.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib2.urlopen(request)

        if response.info().get('Content-Encoding') == 'gzip':
            buf = StringIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read())
            for result in data:

                # retrieving char fragment
                charFragment = result.get('charFragment')
                cfStart = charFragment.get('start')

                # for every charfragment with word find the wiki lemma
                if(cfStart == char_indx):
                    if(result.get('score') >= default_score):
                        default_score = result.get('score')
                        print 'Current score - ',default_score
                        cfEnd = charFragment.get('end')
                        print 'Fragment - ',input[cfStart:cfEnd+1]
                        synsetId = result.get('babelSynsetID')

                        # retrive babelnet lemma
                        lemma_cand = self.retriveLemma(synsetId)
                        print 'lemma - ',lemma_cand


            # TODO: return table wikiId value of final lemma (use self.wiki2id_table)


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




