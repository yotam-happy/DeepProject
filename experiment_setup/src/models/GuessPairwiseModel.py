import random

class GuessPairwiseModel:
    def predict(self, wikilink, candidate1, candidate2):
        if random.randrange(2) == 1:
            return candidate1
        else:
            return candidate2