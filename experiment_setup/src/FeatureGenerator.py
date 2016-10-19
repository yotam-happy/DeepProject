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
        self.mention_features = mention_features
        self.entity_features = entity_features
        self.knockout_model = knockout_model
        self.pointwise_model = pointwise_model
        self.yamada_txt_to_embd = yamada_txt_to_embd

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
        candidates = self._stats.getCandidatesForMention(mention.mention_text())

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
                for cand in candidates:
                    count += self._stats.getCandidatePrior(cand)
                if count == 0:
                    features.append(float(0))
                else:
                    features.append(float(self._stats.getCandidatePrior(entity)) / count)
            else:
                features.append(float(0))
        if 'cond_prior' in self.entity_features:            #P(mention|sense)
            features.append(self._stats.getCandidateConditionalPrior(entity, mention.mention_text()))

        # string similarity features
        page_title = self._db.getPageTitle(entity)
        page_title = utils.text.normalize_unicode(page_title) if page_title is not None else None
        mention = utils.text.normalize_unicode(mention.mention_text())
        if 'entity_title_starts_or_ends_with_mention' in self.entity_features:
            x = 1 if page_title is not None and (page_title.startswith(mention) or page_title.endswith(mention)) else 0
            features.append(x)
        if 'mention_text_starts_or_ends_with_entity' in self.entity_features:
            x = 1 if page_title is not None and (mention.startswith(page_title) or mention.endswith(page_title)) else 0
            features.append(x)
        if 'edit_distance' in self.entity_features:
            features.append(edit_distance(page_title, mention) if page_title is not None else 0)

        # context similarity features
        if 'yamada_context_similarity' in self.entity_features:
            if hasattr(mention.document(), 'yamada_context_embd'):
                context_embd = mention.document().yamada_context_embd
            else:
                txt = ' '.join(mention.document().tokens)
                context_embd = self.yamada_txt_to_embd.text_to_embedding(txt)
                mention.document().yamada_context_embd = context_embd
            self.yamada_txt_to_embd.entity_embd(unicode(self._db.getPageTitle(entity)))

        return features

    def getMentionFeatures(self, mention):
        candidates = self._stats.getCandidatesForMention(mention.mention_text())

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
            cond_mat, votes_mat = self.getCandidateListFeatures(mention, candidates, trunc_param=5)
            features.append(cond_mat.flatten())

        if 'n_candidates' in self.mention_features:
            features.append(len(candidates))

        return features
