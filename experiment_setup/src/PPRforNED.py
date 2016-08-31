import os
import json

import sys


class PPRIterator:

    # new iterator
    def __init__(self, limit_files = 0, path = '/home/noambox/DeepProject/data/PPRforNED/AIDA_candidates', mention_filter=None, resolveIds=False):
        self._path = path
        self._limit_files = limit_files
        self._mention_filter = mention_filter
        # self._stopwords = stopwords.words('english')
        self._resolveIds = resolveIds
        # self._db = db

    def _aida_files(self):
        # iterater over the AIDA candidates files
        for folder in os.listdir(self._path):
            folder_path = os.path.join(self._path ,folder)
            for f in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, f)):
                    continue
                print "opening ", f
                yield open(os.path.join(folder_path, f), 'r')

    def getEntityChunck(self):
        for c, f in enumerate(self._aida_files()):
            for e_chunck in enumerate(self.getEntityChunck(f)):
                yield e_chunck

    @staticmethod
    def getEntityChunckFromFile(f):
        # extracts full ENTITIY and CANDIDATES structure from specified file
        l = f.readlines()
        start_chunck = 0
        end_chunck = 0
        for k, line in enumerate(l):
            if end_chunck > start_chunck:
                start_chunck = end_chunck
            sline = line.split('\t')
            l[k] = sline
            if sline[0] == 'ENTITY' and k is not 0:
                end_chunck = k
                # print 'text: ',l[start_chunck]
                # print start_chunck, 'and ', end_chunck
                yield l[start_chunck:end_chunck]

    def getPharsedEntityChunck(self):
        # returns the entity chuncks for the PPRStatistiscs
        pec = dict() # the phares entity chunck
        for c, f in enumerate(self._aida_files()):
            for e_chunck in enumerate(self.getEntityChunckFromFile(f)):
                entity_details = e_chunck[1][0]
                candidate_details = e_chunck[1][1:]

                # get entity details
                pec['mention'] = entity_details[2][11:] # getting rid of the the field name
                pec['url'] = entity_details[8][4:]
                pec['original'] = entity_details[7][9:]

                # get candidates details
                pec['sensesDetails'] = []
                for ii,cand in enumerate(candidate_details):
                    # the key of each link is the nomarlized Wikititle
                    (pec['sensesDetails']).append({'normWikiTitle':cand[7][11:],'id':cand[1][3:],'inCount':cand[2][8:],'outCount':cand[3][9:],
                                                          'relatedSenses':cand[4][6:].split(';'),'url':cand[5][4:],'normalName':cand[7][11:]})
                # return
                yield pec

            f.close()
            if self._limit_files > 0 and c >= self._limit_files:
                break

class PPRStatistics:

    def __init__(self, _ppr_itr, load_file = None):
        self.ppr_itr = _ppr_itr
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.conceptCounts = dict()
        self.conceptGraph = dict()
        if load_file is not None:
            self.loadFromFile(load_file)

    def calcStatistics(self):
        """
        2. update mentionCounts and mentionLinks
        3. update conceptCounts
        4. update concepGraph = {"id_string": {"inLinks":, "outLinks":, "links":, }}
        Notes: there is no context dictionary
        """
        print "getting statistics"
        for entity_chunck in self.ppr_itr.getPharsedEntityChunck():
            mention_name = entity_chunck['mention'].lower()

            self.mentionCounts[mention_name] = self.mentionCounts.get(mention_name, 0) + 1
            if not mention_name in self.mentionLinks:
                self.mentionLinks[mention_name] = dict()

            # adds links for the gold sense
            gold_sense = entity_chunck['sensesDetails'][0]
            self.mentionLinks[mention_name][gold_sense['id']] = self.mentionLinks[mention_name].get(gold_sense['id'], 0) + 1
            self.conceptCounts[gold_sense['id']] = self.conceptCounts.get(gold_sense['id'], 0) + 1
            self.conceptGraph[gold_sense['id']] = self.conceptGraph.get(gold_sense['id'],
                                                                   {field: gold_sense[field] for field in ('normWikiTitle','inCount', 'outCount', 'relatedSenses', 'url') if field in gold_sense})
            if len(entity_chunck['sensesDetails']) > 1:
                # adds links for other senses
                for sense in entity_chunck['sensesDetails'][1:]:
                    self.mentionLinks[mention_name][sense['id']] = self.mentionLinks[mention_name].get(sense['id'], 0)
                    self.conceptGraph[sense['id']] = self.conceptGraph.get(sense['id'],
                                                                       {field: sense[field] for field in ('normWikiTitle','inCount','outCount','relatedSenses','url') if field in sense} )

        self.prettyPrintStats()

    def saveToFile(self, path):
        """ saves statistics to a file """
        f = open(path, mode='w')
        f.write(json.dumps(self.mentionCounts)+'\n')
        f.write(json.dumps(self.mentionLinks)+'\n')
        f.write(json.dumps(self.conceptCounts)+'\n')
        f.write(json.dumps(self.conceptGraph))
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.conceptCounts = json.loads(l[2])
        self.conceptGraph = json.loads(l[3])
        f.close()

    def prettyPrintStats(self, limit = 5):
        try:
            print 'mentionCounts: ',{k: self.mentionCounts.get(k) for k in self.mentionCounts.keys()[:limit]}
            print 'mentionLinks: ',{k: self.mentionLinks.get(k) for k in self.mentionLinks.keys()[:limit]}
            print 'conceptCounts: ', {k: self.conceptCounts.get(k) for k in self.conceptCounts.keys()[:limit]}
            print 'conceptGraph: ', {k: self.conceptGraph.get(k) for k in self.conceptGraph.keys()[:limit]}
        except :
            print "Unexpected error:", sys.exc_info()[0]

if __name__ == "__main__":

    path = "/home/yotam/pythonWorkspace/deepProject"
    print "Loading iterators+stats..."
    if not os.path.isdir(path):
        path = "/home/noambox/DeepProject"
    elif (not os.path.isdir(path)):
        path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

    ppr_itr = PPRIterator(limit_files = 2, path = path + '/data/PPRforNED/AIDA_candidates')
    ppr_stats = PPRStatistics(ppr_itr)
    ppr_stats.calcStatistics()
    ppr_stats.saveToFile(path + '/data/PPRforNED/ppr_stats')
