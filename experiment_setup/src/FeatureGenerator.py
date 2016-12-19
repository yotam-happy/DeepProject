import json

import numpy as np
import pandas as pd

import UnifyKB
import utils.text
from nltk.metrics.distance import edit_distance
from yamada.text_to_embedding import *

class FeatureGenerator:
    def __init__(self, mention_features={}, entity_features={}, stats=None, db=None, w2v=None,
                 yamada_embedding_path=None,
                 dmodel=None):
        self._stats = stats
        self._db = db
        self._w2v = w2v
        self.mention_features = [x for x in mention_features]
        self.entity_features = [x for x in entity_features]
        print self.entity_features
        self.distance_model = dmodel
        if dmodel is not None:
            self.distance_model_predictor = dmodel.getPredictor()

        if 'yamada_context_similarity' in self.entity_features:
            self.yamada_txt_to_embd = YamadaEmbedder(yamada_embedding_path, db=db) if yamada_embedding_path is not None else None

    def getPointwiseFeatureList(self):
        return self.entity_features + self.mention_features

    def getPairwiseFeatures(self, mention, candidate1, candidate2):
        features_cand1 = self.getEntityFeatures(mention, candidate1)
        features_cand2 = self.getEntityFeatures(mention, candidate2)
        features_mention = self.getMentionFeatures(mention)
        return features_cand1 + features_cand2 + features_mention

    def getPointwiseFeatures(self, mention, entity):
        features_cand1 = self.getEntityFeatures(mention, entity)
        features_mention = self.getMentionFeatures(mention)
        with open('feature_set.txt', 'a') as f:
            f.write(str(entity) + ' ' + str(features_cand1 + features_mention))
        return features_cand1 + features_mention

    def numPairwiseFeatures(self):
        return len(self.entity_features) * 2 + len(self.mention_features)

    def numPointwiseFeatures(self):
        return len(self.entity_features) + len(self.mention_features)

    def getEntityFeatures(self, mention, entity):
        features = []

        page_title = self._db.getPageTitle(entity)
        page_title = utils.text.normalize_unicode(page_title) if page_title is not None else None
        mention_text = utils.text.normalize_unicode(mention.mention_text())

        for feature in self.entity_features:

            # Count features
            if feature == 'prior':
                features.append(self._stats.getCandidatePrior(entity))
            elif feature == 'prior_yamada':
                features.append(self._stats.getCandidatePriorYamadaStyle(entity))
            elif feature == 'normalized_prior':
                features.append(self._stats.getCandidatePrior(entity, normalized=True))
            elif feature == 'normalized_log_prior':
                features.append(self._stats.getCandidatePrior(entity, normalized=True, log=True))
            elif feature == 'relative_prior':
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
            elif feature == 'cond_prior':
                features.append(self._stats.getCandidateConditionalPrior(entity, mention))
            elif feature == 'n_of_candidates':
                features.append(len(mention.candidates))
            elif feature == 'max_prior':
                max_prior = self._stats.getCandidateConditionalPrior(entity, mention)
                for m in mention.document().mentions:
                    if entity in m.candidates and self._stats.getCandidateConditionalPrior(entity, m) > max_prior:
                        max_prior = self._stats.getCandidateConditionalPrior(entity, m)
                features.append(max_prior)

            # string similarity features
            elif feature == 'entity_title_starts_or_ends_with_mention':
                x = 1 if page_title is not None and (page_title.startswith(mention_text) or page_title.endswith(mention_text)) else 0
                features.append(x)
            elif feature == 'mention_text_starts_or_ends_with_entity':
                x = 1 if page_title is not None and (mention_text.startswith(page_title) or mention_text.endswith(page_title)) else 0
                features.append(x)
            elif feature == 'edit_distance':
                features.append(edit_distance(page_title, mention_text) if page_title is not None else 0)

            # context similarity features
            elif feature == 'yamada_context_similarity':
                if not hasattr(mention.document(), 'yamada_context_nouns'):
                    mention.document().yamada_context_nouns = \
                        self.yamada_txt_to_embd.get_nouns(mention.document().sentences)

                if not hasattr(mention.document(), 'yamada_context_embd'):
                    mention.document().yamada_context_embd = dict()
                if mention_text not in mention.document().yamada_context_embd:
                    context_embd = self.yamada_txt_to_embd.text_to_embedding(
                        mention.document().yamada_context_nouns, mention_text)
                    mention.document().yamada_context_embd[mention_text] = context_embd
                context_embd = mention.document().yamada_context_embd[mention_text]
                entity_embd = self.yamada_txt_to_embd.from_the_cache(entity)
                if entity_embd is not None:
#                    print self.yamada_txt_to_embd.similarity(context_embd, entity_embd)
                    features.append(self.yamada_txt_to_embd.similarity(context_embd, entity_embd))
                else:
                    #print 0
                    features.append(0.0)
            elif feature == 'our_context_similarity':
                if not hasattr(mention.document(), 'our_context_nouns'):
                    mention.document().our_context_nouns = \
                        self._w2v.get_nouns(mention.document().sentences)

                if not hasattr(mention.document(), 'our_context_embd'):
                    mention.document().our_context_embd = dict()
                if mention_text not in mention.document().our_context_embd:
                    context_embd = self._w2v.text_to_embedding(
                        mention.document().our_context_nouns, mention_text)
                    mention.document().our_context_embd[mention_text] = context_embd
                context_embd = mention.document().our_context_embd[mention_text]
                entity_embd = self._w2v.get_entity_vec(entity)
                if entity_embd is not None:
                    print self._w2v.similarity(context_embd, entity_embd)
                    features.append(self._w2v.similarity(context_embd, entity_embd))
                else:
                    print 0
                    features.append(0.0)
            elif feature == 'distance_model':
                x = self.distance_model_predictor.predict_prob(mention, entity)
                features.append(x)
            else:
                raise "feature undefined"


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
