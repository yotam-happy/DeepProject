import json

import numpy as np
import pandas as pd

import UnifyKB
import utils.text
from nltk.metrics.distance import edit_distance

class FeatureGenerator:
    def __init__(self, mention_features={}, entity_features={}, stats=None, db=None,
                 knockout_model=None,
                 pointwise_model=None,
                 yamada_txt_to_embd=None):
        self._stats = stats
        self._db = db
        self.mention_features = [x for x in mention_features]
        self.entity_features = [x for x in entity_features]
        self.knockout_model = knockout_model
        self.pointwise_model = pointwise_model
        self.yamada_txt_to_embd = yamada_txt_to_embd

    def getPointwiseFeatureList(self):
        return self.entity_features + self.mention_features

    def getCandidateListFeatures(self, mention, candi_list, trunc_param=5):
        winner, cond_prob, cond_votes = \
                self.knockout_model.predict2(mention, candidates=candi_list, returnProbMode=True)
        if trunc_param is not None:
            cond_prob, cond_votes, sort_indx_cand = self.sortAndTruncate(cond_prob, cond_votes, trunc_param)
        else:
            return cond_prob, cond_votes
        return cond_prob, cond_votes, sort_indx_cand

    @staticmethod
    def sortAndTruncate(prob_mat, votes_mat, trunc_survivors=None):
        votes_summary = np.sum(votes_mat, axis=1)
        sort_index = np.flipud(np.argsort(votes_summary))
        sorted_votes_mat = votes_mat[sort_index].T[sort_index].T
        sorted_prob_mat = prob_mat[sort_index].T[sort_index].T
        if trunc_survivors > len(votes_mat):
            votes_mat = sorted_votes_mat[:trunc_survivors, :trunc_survivors]
            prob_mat = sorted_prob_mat[:trunc_survivors, :trunc_survivors]
        elif trunc_survivors is not None:
            # TODO: is it right to padd here by zeros?
            pad_size = trunc_survivors - len(votes_mat)
            votes_mat = np.lib.pad(sorted_votes_mat, (0, pad_size), 'constant', constant_values=(0, 0))
            prob_mat = np.lib.pad(sorted_prob_mat, (0, pad_size), 'constant', constant_values=(0, 0))
        else:
            return sorted_prob_mat, sorted_votes_mat, sort_index
        return prob_mat, votes_mat, sort_index

    def getPairwiseFeatures(self, mention, candidate1, candidate2):
        features_cand1 = self.getEntityFeatures(mention, candidate1)
        features_cand2 = self.getEntityFeatures(mention, candidate2)
        features_mention = self.getMentionFeatures(mention)
        return features_cand1 + features_cand2 + features_mention

    def getPointwiseFeatures(self, mention, entity):
        features_cand1 = self.getEntityFeatures(mention, entity)
        features_mention = self.getMentionFeatures(mention)
        return features_cand1 + features_mention

    def numPairwiseFeatures(self):
        return len(self.entity_features) * 2 + len(self.mention_features)

    def numPointwiseFeatures(self):
        return len(self.entity_features) + len(self.mention_features)

    def getEntityFeatures(self, mention, entity):
        features = []

        for feature in self.entity_features:
            # Count features
            if feature == 'prior':
                features.append(self._stats.getCandidatePrior(entity))
            if feature == 'prior_yamada':
                features.append(self._stats.getCandidatePriorYamadaStyle(entity))
            if feature == 'normalized_prior':
                features.append(self._stats.getCandidatePrior(entity, normalized=True))
            if feature == 'normalized_log_prior':
                features.append(self._stats.getCandidatePrior(entity, normalized=True, log=True))
            if feature == 'relative_prior':
                if entity in mention.candidates:
                    count = 0
                    for cand in mention.candidates:
                        count += self._stats.getCandidatePrior(cand)
                    if count == 0:
                        features.append(float(0))
                    else:
                        features.append(float(self._stats.getCandidatePrior(entity)) / count)
                else:
                    features.append(float(0))
            if feature == 'cond_prior':
                features.append(self._stats.getCandidateConditionalPrior(entity, mention))
            if feature == 'n_of_candidates':
                features.append(len(mention.candidates))
            if feature == 'max_prior':
                max_prior = self._stats.getCandidateConditionalPrior(entity, mention)
                for m in mention.document().mentions:
                    if entity in m.candidates and self._stats.getCandidateConditionalPrior(entity, m) > max_prior:
                        max_prior = self._stats.getCandidateConditionalPrior(entity, m)
                features.append(max_prior)

            # string similarity features
            page_title = self._db.getPageTitle(entity)
            page_title = utils.text.normalize_unicode(page_title) if page_title is not None else None
            mention_text = utils.text.normalize_unicode(mention.mention_text())
            if feature == 'entity_title_starts_or_ends_with_mention':
                x = 1 if page_title is not None and (page_title.startswith(mention_text) or page_title.endswith(mention_text)) else 0
                features.append(x)
            if feature == 'mention_text_starts_or_ends_with_entity':
                x = 1 if page_title is not None and (mention_text.startswith(page_title) or mention_text.endswith(page_title)) else 0
                features.append(x)
            if feature == 'edit_distance':
                features.append(edit_distance(page_title, mention_text) if page_title is not None else 0)

            # context similarity features
            if feature == 'yamada_context_similarity':
                if not hasattr(mention.document(), 'yamada_context_embd'):
                    txt = ' '.join(mention.document().tokens)
                    context_embd = self.yamada_txt_to_embd.text_to_embedding(txt)
                    mention.document().yamada_context_embd = context_embd
                context_embd = mention.document().yamada_context_embd
                entity_embd = self.yamada_txt_to_embd.entity_embd(unicode(self._db.getPageTitle(entity)))

        return features

    def getMentionFeatures(self, mention):
        features = []

        for feature in self.mention_features:
            if feature == 'max_prior':
                if not hasattr(mention.document(), 'max_prior'):
                    max_prior = 0
                    for m in mention.document().mentions:
                        for entity in m.candidates:
                            p = self._stats.getCandidateConditionalPrior(entity, mention)
                            max_prior = p if p > max_prior else max_prior
                    mention.document().max_prior = max_prior
                features.append(mention.document().max_prior)
        return features
