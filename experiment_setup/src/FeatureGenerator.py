import json

import numpy as np
import pandas as pd

import UnifyKB
import utils.text
from nltk.metrics.distance import edit_distance

class FeatureGenerator:
    def __init__(self, mention_features={}, entity_features={}, stats=None, db=None, knockout_model=None, pointwise_model=None):
        self._stats = stats
        self._db = db
        self.mention_features = mention_features
        self.entity_features = entity_features
        self.knockout_model = knockout_model
        self.pointwise_model = pointwise_model

    def getCandidateListFeatures(self, wlink, candi_list, trunc_param = 5):
        winner, cond_prob, cond_votes = \
                self.knockout_model.predict2(wlink, candidates = candi_list, returnProbMode = True)
        if trunc_param is not None:
            cond_prob, cond_votes, sort_indx_cand = self.sortAndTruncate(cond_prob, cond_votes, trunc_param)
        else:
            return cond_prob, cond_votes
        return cond_prob, cond_votes, sort_indx_cand

    @staticmethod
    def sortAndTruncate(prob_mat, votes_mat, trunc_survivors = None):
        votes_summary = np.sum(votes_mat, axis = 1)
        sort_index = np.flipud(np.argsort(votes_summary))
        sorted_votes_mat = votes_mat[sort_index].T[sort_index].T
        sorted_prob_mat = prob_mat[sort_index].T[sort_index].T
        if trunc_survivors > len(votes_mat):
            votes_mat = sorted_votes_mat[:trunc_survivors, :trunc_survivors]
            prob_mat = sorted_prob_mat[:trunc_survivors, :trunc_survivors]
        elif trunc_survivors is not None:
            # TODO: is it right to padd here by zeros?
            pad_size = trunc_survivors - len(votes_mat)
            votes_mat = np.lib.pad(sorted_votes_mat, (0, pad_size), 'constant', constant_values=(0,0))
            prob_mat = np.lib.pad(sorted_prob_mat, (0, pad_size), 'constant', constant_values=(0, 0))
        else:
            return sorted_prob_mat, sorted_votes_mat, sort_index
        return prob_mat, votes_mat, sort_index

    def getPairwiseFeatures(self, wlink, candidate1, candidate2):
        features_cand1 = self.getEntityFeatures(wlink, candidate1)
        features_cand2 = self.getEntityFeatures(wlink, candidate2)
        features_mention = self.getMentionFeatures(wlink)
        return features_cand1 + features_cand2 + features_mention

    def getPointwiseFeatures(self, wlink, entity):
        features_cand1 = self.getEntityFeatures(wlink, entity)
        features_mention = self.getMentionFeatures(wlink)
        return features_cand1 + features_mention

    def numPairwiseFeatures(self):
        return len(self.entity_features) * 2 + len(self.mention_features)

    def numPointwiseFeatures(self):
        return len(self.entity_features) + len(self.mention_features)

    def getEntityFeatures(self, wlink, entity):
        candidates = self._stats.getCandidatesForMention(wlink["word"])

        features = []

        # Count features
        if 'prior' in self.entity_features:
            features.append(self._stats.getCandidatePrior(entity))
        if 'normalized_prior' in self.entity_features:
            features.append(self._stats.getCandidatePrior(entity, normalized=True))
        if 'normalized_log_prior' in self.entity_features:
            features.append(self._stats.getCandidatePrior(entity, normalized=True, log=True))
        if 'relative_prior' in self.entity_features:
            if entity in candidates:
                count = 0
                for cand, c in candidates.items():
                    count += self._stats.getCandidatePrior(cand)
                if count == 0:
                    features.append(float(0))
                else:
                    features.append(float(self._stats.getCandidatePrior(entity)) / count)
            else:
                features.append(float(0))
        if 'cond_prior' in self.entity_features:            #P(mention|sense)
            features.append(self._stats.getCandidateConditionalPrior(entity, wlink["word"]))

        # string similarity features
        page_title = self._db.getPageTitle(entity)
        page_title = utils.text.normalize_unicode(page_title) if page_title is not None else None
        mention = utils.text.normalize_unicode(wlink["word"])
        if 'entity_title_starts_or_ends_with_mention':
            x = 1 if page_title is not None and (page_title.startswith(mention) or page_title.endswith(mention)) else 0
            features.append(x)
        if 'mention_text_starts_or_ends_with_entity':
            x = 1 if page_title is not None and (mention.startswith(page_title) or mention.endswith(page_title)) else 0
            features.append(x)
        if 'edit_distance':
            features.append(edit_distance(page_title, mention) if page_title is not None else 0)

        return features

    def getMentionFeatures(self, wlink):
        candidates = self._stats.getCandidatesForMention(wlink["word"])

        features = []

        # Count features
        if 'yamadas_base' in self.mention_features:
            # returns Yamadas local base features. The doc features are calculated differently.
            # here an optional sense is a candidate (represented as 'e' for entity in Yamada)
            for i, cand in enumerate(candidates.items()):
                features.append(self._stats.getCandidateProbabilityYamadaStyle(str(cand[0]))) # entity prior Prob(sense)
                features.append(cand[1]) # conditional entity prior Prob(sense|mention)
            features.append(len(candidates)) # number of entity candidates (s) for mention m

        if 'rnn_model_feature_local' in self.mention_features:
            # getting the matrixes of conditional prob and ranking FIXME: flatten matrix and solve trunc_param issue
            cond_mat, votes_mat = self.getCandidateListFeatures(wlink, candidates, trunc_param=5)
            features.append(cond_mat.flatten())

        if 'n_candidates' in self.mention_features:
            features.append(len(candidates))

        return features

    def getMentionListFeatures(self,wlink_list, pointwise_feature = True):
        """
        get a mention list and extracts features related each mention and the source doc
        that all mentions came from.
        :param wlink_list: mention list
        :return: feature vector of all mention and candidates features flattened for mention (yield)
        """
        features = []

        if 'yamadas_base' in self.mention_features:
            doc_entity_prior = dict()
            for wlink in wlink_list:
                doc_entity_prior = UnifyKB.updateIntersection(self._stats.getCandidatesForMention(wlink['word']),doc_entity_prior)
                # print "doc entity prior: ", self._stats.getCandidatesForMention(wlink['word'])
             # if doc_entity_prior is not:
            for wlink in wlink_list:
                features, candidates = self.getMentionFeatures(wlink)
                feature_mat = np.reshape(np.asarray(features[:-1]), (-1, 2))
                feature_df = pd.DataFrame(feature_mat, index=candidates.keys(), columns=["P(s)","P(s|m)"])
                feature_df['max_entity_prior'] = [doc_entity_prior[idx] for idx in feature_df.index]

                # up to here we have the full base structure in a matrix except of the n_candidates feature
                if 'rnn_model_feature' in self.mention_features:
                    # candidates = self._stats.getCandidatesForMention(wlink['word'])
                    if pointwise_feature:
                        # taking the mean and variance of the a_beats_b matrix and the normalized sum of votes
                        # print 'generating model features'
                        num_cands = len(candidates.keys())
                        if num_cands > 1:
                            cond_mat, votes_mat = self.getCandidateListFeatures(wlink, candidates, trunc_param=None)
                            # print 'concat model features'
                            mean_cond = []; var_cond = []; votes_prec = []
                            for row in xrange(len(candidates.keys())):
                                range_ind = filter(lambda x: x != row, np.arange(num_cands))
                                mean_cond.append(np.mean(cond_mat[row,range_ind ]))
                                var_cond.append(np.var(cond_mat[row, range_ind]))
                                votes_prec.append( np.sum(votes_mat[row, range_ind]) / float((num_cands-1) * 2))

                        elif num_cands == 1:
                            mean_cond = 1
                            var_cond = 0
                            votes_prec = 1

                        feature_df['mean_model_prob'] = mean_cond
                        feature_df['var_model_score'] = var_cond
                        feature_df['votes_model_score'] = votes_prec


                    else:
                        # this entire line concatinates the candidate features to the exsisting base so the eventually we have for each candidate -
                        # features_mat[candidate = s] = [prob(s) prob(s|m) max_prob(s|{m_in_doc}) cond_prob(s) votes_(s)]. This highly depends on the
                        # truncation_number - the number of candidates that are in the models output matrices
                        cond_mat, votes_mat, sort_indx = self.getCandidateListFeatures(wlink, candidates, trunc_param=5)
                        feature_mat = np.asanyarray(
                                       [np.concatenate((feature_mat[sort_indx][row,:],cond_mat[row,:],votes_mat[row,:]))
                                       if (row < len(cond_mat)) is True else
                                       np.concatenate((feature_mat[sort_indx][row, :], np.zeros_like(cond_mat[0, :]), np.zeros_like(cond_mat[0, :])))
                                       for row in xrange(len(candidates))])

                # feature_wlink = np.append(feature_mat.flatten(), features[-1]) # flatten and add last base feature #{candidates}
                feature_df['#s'] = features[-1] * np.ones(feature_df.shape[0])
                feature_df['label'] = 2 * (np.asarray(feature_df.index) == float(wlink['wikiId'])) - 1
                yield feature_df

    def save(self, fname):
        f = open(fname, mode='w')
        f.write(json.dumps({x: 1 for x in self.mention_features}) + '\n')
        f.write(json.dumps({x: 1 for x in self.entity_features}) + '\n')
        f.close()

    def load(self, fname):
        f = open(fname, mode='r')
        l = f.readlines()
        self.mention_features = json.loads(l[0]).keys()
        self.entity_features = json.loads(l[1]).keys()
        f.close()
