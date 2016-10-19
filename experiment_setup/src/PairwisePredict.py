import itertools
import math
import operator
import random

from WikilinksStatistics import *


class PairwisePredict:
    """
    This model takes a pairwise model that can train/predict on pairs of candidates for a wikilink
    and uses it to train/predict on a list candidates using a knockout method.
    """

    def __init__(self, pairwise_model):
        """
        :param pairwise_model:  The pairwise model used to do prediction/training on a triplet
                                (wikilink,candidate1,candidate2)
        """
        self._pairwise_model = pairwise_model

    def predict(self, mention):
        # do a knockout
        l = [candidate for candidate in mention.candidates]
        random.shuffle(l)
        return self._predict(mention, l)

    def predict2(self, mention, returnProbMode = False):
        """
        pairwise prediction between all possible pairs of candidates (no self pairs)
        every comprison is calculated twice for eliminating order importance
        :param wikilink:
        :param candidates:
        :param returnProbMode: if true returns also vote matrix (i rows j column matrix with number of votes of
                                i beats j. Returns also cond_prob (Same idea with conditional probability of i beats j)
        :return:
        """
        l = [candidate for candidate in mention.candidates]
        if len(l) == 1:
            return l[0]

        cond_prob = np.ones((len(mention.candidates), len(mention.candidates)))
        cond_votes = np.zeros((len(mention.candidates), len(mention.candidates)))
        ranking = {x:0.0 for x in l}

        # by using a and b we diminish the importance of order in the input
        for i in xrange(len(l) - 1):
            for j in xrange(i + 1, len(l)):
                if returnProbMode:
                    a, i_beats_j_1 , j_beats_i_1, votes_i_1, votes_j_1 = \
                        self.getWinnerProbAndUpdateVotes(mention, l[i], l[j] , cond_votes[i][j], cond_votes[j][i])
                    b, j_beats_i_2, i_beats_j_2, votes_j_2, votes_i_2 = \
                        self.getWinnerProbAndUpdateVotes(mention, l[j], l[i], cond_votes[j][i], cond_votes[i][j])
                    if a and b is not None:
                        cond_votes[i][j], cond_votes[j][i] = votes_i_1 + votes_i_2, votes_j_1 + votes_j_2
                    else:
                        cond_votes[i][j] = cond_votes[j][i] = None # TODO : verify that the none task is handled right

                    cond_prob[i][j] = sum(filter(None, [i_beats_j_1, i_beats_j_2]))
                    cond_prob[i][j] *= 0.5 if cond_prob[i][j] is not None else 0
                    cond_prob[j][i] = sum(filter(None, [j_beats_i_1, j_beats_i_2]))
                    cond_prob[j][i] *= 0.5 if cond_prob[i][j] is not None else 0
                else:
                    a = self._pairwise_model.predict(mention, l[i], l[j])
                if a is not None:
                    ranking[a] += 1
                b = self._pairwise_model.predict(mention, l[j], l[i])
                if b is not None:
                    ranking[b] += 1

        m = max(ranking.iteritems(), key=operator.itemgetter(1))[0]
        mv = max(ranking.iteritems(), key=operator.itemgetter(1))[1]
        if m == 0:
            return None
        finals = {x: mention.candidates[x] for x,y in ranking.items() if y == mv}
        final = max(finals.iteritems(), key=operator.itemgetter(1))[0]
        if returnProbMode:
            # print 'candidates order: ',l
            # print 'cond_votes: ',filter(None, cond_votes.tolist())
            final = l[np.argmax(np.sum(filter(None, cond_votes.tolist()), axis=1))]
            return final, cond_prob, cond_votes
        else:
            return final

    def getWinnerProbAndUpdateVotes(self, mention, cand_first, cand_last, a_beats_b, b_beats_a):
        try:
            winner , first_cand_winner_prob = self._pairwise_model.predict(mention, cand_first, cand_last,
                                                                           return_score=True)
        except:
            print 'wlink: ', mention.mention_text(), '\t first: ', cand_first, '\t last: ', cand_last # FIXME

        if winner is None:
            return None, None, None, None, None
        else:
            second_cand_winner_prob = 1 - first_cand_winner_prob
            if winner == cand_first and winner is not None:
                a_beats_b += 1
            elif winner is not None:
                b_beats_a += 1
            # print '** prob of ',str(cand_first),' to beat  ',str(cand_last),' is: ', str(first_cand_winner_prob)
            # print '** votes ',str(cand_first),' beats ',str(cand_last),' : ',str(a_beats_b)
            # print 'winner :', winner,'\n'
            return winner, first_cand_winner_prob, second_cand_winner_prob, a_beats_b, b_beats_a

    def _predict(self, mention, l):
        while len(l) > 1:
            # create a list of surviving candidates by comparing couples
            next_l = []

            for i in range(0, len(l) - 1, 2):
                pr = self._pairwise_model.predict(mention, l[i], l[i+1])
                a = l[i] if pr > 0.5 else l[i+1]
                if a is not None:
                    next_l.append(a)

            if len(l) % 2 == 1:
                next_l.append(l[-1])
            l = next_l

        if len(l) == 0:
            return None
        return l[0]
