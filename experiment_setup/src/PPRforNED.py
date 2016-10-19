import json
import os
import sys

import numpy as np
import math

class PPRIterator:

    # new iterator
    def __init__(self, limit_files = 0, path = '/home/noambox/DeepProject/data/PPRforNED/AIDA_candidates', mention_filter=None, resolveIds=False):
        self._path = path
        self._limit_files = limit_files
        self._mention_filter = mention_filter
        self._resolveIds = resolveIds

    def _aida_files(self):
        # iterater over the AIDA candidates files
        for folder in os.listdir(self._path):
            folder_path = os.path.join(self._path ,folder)
            for f in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, f)):
                    continue
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
        for k, line in enumerate(l):
            sline = line.split('\t')
            l[k] = sline
            if sline[0] == 'ENTITY' and k is not 0:
                # print 'text: ',l[start_chunck]
                # print start_chunck, 'and ', end_chunck
                yield l[start_chunck:k]
                start_chunck = k

        # funny... one of the files is empty
        if len(l) > start_chunck:
            yield l[start_chunck:]

    def getPharsedEntityChunck(self):
        # returns the entity chuncks for the PPRStatistiscs
        pec = dict() # the phares entity chunck
        for c, f in enumerate(self._aida_files()):
            for e_chunck in enumerate(self.getEntityChunckFromFile(f)):
                entity_details = e_chunck[1][0]
                candidate_details = e_chunck[1][1:]

                # get entity details
                pec['mention'] = entity_details[7][9:] # getting rid of the the field name
                pec['url'] = entity_details[8][4:]
                pec['original'] = entity_details[7][9:]

                # get candidates details
                pec['sensesDetails'] = []
                for ii,cand in enumerate(candidate_details):
                    # the key of each link is the nomarlized Wikititle
                    (pec['sensesDetails']).append({'normWikiTitle':cand[7][11:],'id':int(cand[1][3:]),'inCount':int(cand[2][8:]),'outCount':int(cand[3][9:]),
                                                          'relatedSenses':cand[4][6:].split(';'),'url':cand[5][4:],'normalName':cand[7][11:]})
                # return
                yield pec

            f.close()
            if (not self._limit_files == 0) and c >= self._limit_files:
                break

class PPRStatistics:

    def __init__(self, _ppr_itr, load_file = None):
        self.ppr_itr = _ppr_itr
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.mentionLinksUrl = dict()
        self.conceptCounts = dict()
        if load_file is not None:
            self.loadFromFile(load_file)
        self.conceptCountsSum = sum(self.conceptCounts.values())
        self.conceptLogCountsVariance = np.var([math.log(float(x) + 1) for x in self.conceptCounts.values()])
        self.conceptCountsVariance = np.var([float(x) for x in self.conceptCounts.values()])

    def _sortedList(self, l):
        l = [(k,v) for k,v in l.items()]
        l.sort(key=lambda (k,v):-v)
        return l

    def getCandidateConditionalPrior(self, concept, mention):
        raise "not supported"

    def getCandidatePrior(self, concept, normalized=False, log=False):
        if concept not in self.conceptCounts:
            return 0

        if not normalized:
            return float(self.conceptCounts[concept]) / self.conceptCountsSum

        # if normalized, normalize by variance
        if log:
            return math.log(float(self.conceptCounts[concept]) + 1) / self.conceptLogCountsVariance \
                if concept in self.conceptCounts else 0
        else:
            return float(self.conceptCounts[concept]) / self.conceptCountsVariance \
                if concept in self.conceptCounts else 0

    def getCandidatesForMention(self, mention, p=0.01, t=5):
        """
        Returns the most probable sense + all other candidates where p(candidate|mention)>=p
        and with at least t appearances

        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        mention = mention.lower()
        if mention not in self.mentionLinks or len(self.mentionLinks[mention]) == 0:
            return {}

        l = self._sortedList(self.mentionLinks[mention])
        tot = sum([x[1] for x in l])
        out = dict()
        for x in l:
            if len(out) == 0 or tot == 0 or (float(x[1]) / tot >= p and x[1] > t):
                out[int(x[0])] = x[1] if tot != 0 else 1

        # now calc actual priors
        tot = sum([x for x in out.values()])
        if tot == 0:
            out = {int(x): 1.0 / len(out) for x, y in out.iteritems()}
        else:
            out = {int(x): float(y)/tot for x, y in out.iteritems()}
        return out

    def getMostProbableSense(self, mention):
        cands = self.getCandidateUrlsForMention(mention)
        return max(cands.iterkeys(), key=(lambda key: cands[key]))

    def getMostProbableSense2(self, cands):
        cands = {x: self.conceptCounts[str(x)] if str(x) in self.conceptCounts else 0 for x in cands}
        if len(cands) == 0:
            return None
        return max(cands.iterkeys(), key=(lambda key: cands[key]))

    def getCandidateProbabilityYamadaStyle(self,concept): # TODO: change wikistats to same code...
        counter = 0.0
        for links in self.mentionLinks.iteritems():
            if concept in links[1].keys():
                counter+=1.0
        return float(counter)/ len(self.mentionCounts)


    def getCandidateUrlsForMention(self, mention, p=0.01, t=5):
        """
        Returns the most probable sense + all other candidates where p(candidate|mention)>=p
        and with at least t appearances

        :param mention:     the mention to search for
        :return:            returns a dictionary: (candidate,count)
        """
        mention = mention.lower()
        if mention not in self.mentionLinksUrl:
            return {}

        l = self._sortedList(self.mentionLinksUrl[mention])
        tot = sum([x[1] for x in l])
        out = dict()
        for x in l:
            if len(out) == 0 or tot == 0 or (float(x[1]) / tot >= p and x[1] > t):
                out[x[0]] = x[1] if tot != 0 else 1

        # now calc actual priors
        tot = sum([x for x in out.values()])
        if tot == 0:
            tot = 1
        out = {x: float(y)/tot for x, y in out.iteritems()}
        return out

    def calcStatistics(self):
        self.mentionCounts = dict()
        self.mentionLinks = dict()
        self.mentionLinksUrl = dict()
        self.conceptCounts = dict()
        print "getting statistics"
        for entity_chunck in self.ppr_itr.getPharsedEntityChunck():
            mention_name = entity_chunck['mention'].lower()

            self.mentionCounts[mention_name] = self.mentionCounts.get(mention_name, 0) + 1
            if mention_name not in self.mentionLinks:
                self.mentionLinks[mention_name] = dict()
                self.mentionLinksUrl[mention_name] = dict()

            for sense in entity_chunck['sensesDetails']:
                self.mentionLinks[mention_name][sense['id']] = sense['inCount']
                self.mentionLinksUrl[mention_name][sense['url']] = sense['inCount']
                self.conceptCounts[sense['id']] = sense['inCount']

        self.prettyPrintStats()

    def saveToFile(self, path):
        """ saves statistics to a file """
        f = open(path, mode='w')
        f.write(json.dumps(self.mentionCounts)+'\n')
        f.write(json.dumps(self.mentionLinks)+'\n')
        f.write(json.dumps(self.mentionLinksUrl)+'\n')
        f.write(json.dumps(self.conceptCounts)+'\n')
        f.close()

    def loadFromFile(self, path):
        """ loads statistics from a file """
        f = open(path, mode='r')
        l = f.readlines()
        self.mentionCounts = json.loads(l[0])
        self.mentionLinks = json.loads(l[1])
        self.mentionLinksUrl = json.loads(l[2])
        self.conceptCounts = json.loads(l[3])
        f.close()

    def prettyPrintStats(self, limit=5):
        try:
            print 'mentionCounts: ', {k: self.mentionCounts.get(k) for k in self.mentionCounts.keys()[:limit]}
            print 'mentionLinks: ', {k: self.mentionLinks.get(k) for k in self.mentionLinks.keys()[:limit]}
            print 'conceptCounts: ', {k: self.conceptCounts.get(k) for k in self.conceptCounts.keys()[:limit]}
        except:
            print "Unexpected error:", sys.exc_info()[0]

#if __name__ == "__main__":
#
#    path = "/home/yotam/pythonWorkspace/deepProject"
#    print "Loading iterators+stats..."
#    if not os.path.isdir(path):
#        path = "/home/noambox/DeepProject"
#    elif not os.path.isdir(path):
#        path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"
#
#    ppr_itr = PPRIterator(path=path + '/data/PPRforNED/AIDA_candidates')
#    ppr_stats = PPRStatistics(ppr_itr)
#    ppr_stats.calcStatistics()
#    ppr_stats.saveToFile(path + '/data/PPRforNED/ppr_stats')