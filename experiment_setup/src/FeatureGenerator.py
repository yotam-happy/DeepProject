import json


class FeatureGenerator:
    def __init__(self, mention_features={}, entity_features={}, stats=None, db=None):
        self._stats = stats
        self._db = db
        self.mention_features = mention_features
        self.entity_features = entity_features

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
        if 'log_prior' in self.entity_features:
            features.append(self._stats.getConceptPrior(entity, log=True))
        if 'prior' in self.entity_features:
            features.append(self._stats.getConceptPrior(entity))
        if 'relative_prior' in self.entity_features:
            if entity in candidates:
                count = 0
                for cand, c in candidates.items():
                    count += self._stats.getConceptPrior(cand)
                features.append(float(self._stats.getConceptPrior(entity)) / count)
        if 'cond_prior' in self.entity_features:
            features.append(candidates[entity] if entity in candidates else 0)
        return features

    def getMentionFeatures(self, wlink):
        candidates = self._stats.getCandidatesForMention(wlink["word"])

        features = []

        # Count features
        if 'n_candidates' in self.mention_features:
            features.append(len(candidates))

        return features

    def save(self, fname):
        f = open(fname, mode='w')
        f.write(json.dumps(self.mention_features) + '\n')
        f.write(json.dumps(self.entity_features) + '\n')
        f.close()

    def load(self, fname):
        f = open(fname, mode='r')
        l = f.readlines()
        self.mention_features = json.loads(l[0])
        self.entity_features = json.loads(l[1])
        f.close()
