import json
import os

import numpy as np

'''
This file enables unifying several KB into 1 single statistics KB.
It takes wikilink, intralink, PPRforNED and produces a single KB based on those
'''

# pprforned structure
#         f.write(json.dumps(self.mentionCounts)+'\n')
#         f.write(json.dumps(self.mentionLinks)+'\n')
#         f.write(json.dumps(self.mentionLinksUrl)+'\n')
#         f.write(json.dumps(self.conceptCounts)+'\n')

# wikistatistisc structure
#         f.write(json.dumps(self.mentionCounts)+'\n')
#         f.write(json.dumps(self.mentionLinks)+'\n')
#         f.write(json.dumps(self.conceptCounts)+'\n')
#         f.write(json.dumps(self.contextDictionary)+'\n')
#         f.write(json.dumps(self.seenWith))
#     mentionCounts       dictionary of mention=count. Where mention is a surface term to be disambiguated
#                         and count is how many times it was seen n the dataset
#     conceptCounts       dictionary of concept=count. Where a concept is a wikipedia id (a sense), and count
#                         is how many times it was seen (how many mentions refered to it)
#     contextDictionary   dictionary of all words that appeared inside some context (and how many times)
#     mentionLinks        holds for each mention a dictionary of conceptsIds it was reffering to and how many
#                         times each. (So its a dictionary of dictionaries)

def FullKnowledgeBase(path_list, dest_path = os.getcwd(), noPPR = False):
    """
    create KB of all available Kb. It takes for each mutual mention, the mentionList. The mentionList is being updated
    and a new mentionCounts is derived.
    :return:
    """
    mentionCounts= dict()
    mentionLinks = dict()
    conceptCounts = dict()
    if noPPR:
        contextDictionary = dict()
        seenWith = dict()

    for stat_path in path_list:
        l = f.readlines()
        f = open(stat_path, mode='r')
        mentionCounts_itr = json.loads(l[0])
        mentionLinks_itr = json.loads(l[1])
        conceptCounts_itr = json.loads(l[3])
        if noPPR:
            contextDictionary_itr = json.loads(l[3])
            conceptCounts_itr = json.loads(l[3])

        # TODO: change intersec criteria to expand mention links and recount mentionCount (check if the conceptID are the same for intra, wikilink and ppr)
        # TODO: do the same for seenwith
        # TODO: in conceptCounts take simple the one with higher score

        update_out = updateIntersection(mentionCounts, mentionCounts_itr)
        for i,k in enumerate(update_out['intrsc_keys']):
            if update_out['map'][i]:
                mentionCounts[k] = mentionCounts_itr[k]
                mentionLinks[k] = mentionLinks_itr[k]
                conceptCounts[k] = conceptCounts_itr[k]
                if noPPR:
                    contextDictionary = dict()
                    seenWith = dict()

        mentionCounts = mergeDict(mentionCounts_itr, mentionCounts)
        mentionLinks = mergeDict(mentionLinks_itr, mentionLinks)
        conceptCounts = mergeDict(conceptCounts_itr, conceptCounts)
        f.close()

    # saves statistics to a file
    f = open(dest_path, mode='w')
    f.write(json.dumps(mentionCounts)+'\n')
    f.write(json.dumps(mentionLinks)+'\n')
    f.write(json.dumps(conceptCounts)+'\n')
    if noPPR:
        f.write(json.dumps(contextDictionary)+'\n')
        f.write(json.dumps(seenWith))

    f.close()
##
def updateIntersection( a, b):
    """
    takes the intersection of 2 dict and updates the first's intersected fields (a) according to the higher values
    :param a:   mentionCounts dict()
    :param b:   mentionCounts_itr dict()
    :return: out structure with updated input structs
    """
    intrsc_keys = set(a.keys()) & set(b.keys())
    a_intr_val = [a[k] for k in intrsc_keys]
    b_intr_val = [b[k] for k in intrsc_keys]
    max_ind = np.argmax(np.asarray([a_intr_val, b_intr_val]),0)
    return {'map': max_ind, 'intrsc_keys':intrsc_keys}
##
def mergeDict(a, b):
    """
    merges dicts when b overrides key values that are mutual with a
    :param a:
    :param b:
    :return: z
    """
    z = a.copy()
    z.update(b)
    return z
##


if __name__ == "__main__":

    path = "/home/yotam/pythonWorkspace/deepProject"
    print "Loading iterators+stats..."
    if not os.path.isdir(path):
        path = "/home/noambox/DeepProject"
    elif (not os.path.isdir(path)):
        path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

    wlink_stats_path = path + "/data/wikilinks/all/train-stats"
    intra_stats_path = path + "/data/intralinks/train-stats"
    ppr_stats_path = path + "/data/PPRforNED/ppr_stats"

    # create KB of all available Kb
    FullKnowledgeBase([wlink_stats_path, intra_stats_path, ppr_stats_path], dest_path = path+"/data/full-stats", noPPR= False)

    # create KB of wikilinks and intralinks
    FullKnowledgeBase([wlink_stats_path, intra_stats_path], dest_path = path+"/data/intra-wlink-stats", noPPR= True)

    # 3. create DS of intra and wikilinks  ( train , test, eval )

