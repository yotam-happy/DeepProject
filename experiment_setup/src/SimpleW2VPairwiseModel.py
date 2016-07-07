from Word2vecLoader import *


class SimpleW2VPairwiseModel:
    """
    This is a simple model that decides which of two candidates is better by simply comparing
    the distance between the w2v vectors of each of the candidates to the w2v vector of the context
    The w2v vector for the context is calculated simply as the average of all context words
    """

    def __init__(self, w2v):
        self._w2v = w2v

    def predict(self, wikilink, candidate1, candidate2):
        """
        The prediction function of a similarity model
        :return:    returns one of the two candidates - the one deemed better
                    This function can returns None if it has nothing to say on either of the candidates
                    this is considered as saying they are both very unlikely and should be eliminated
        """

        if candidate1 not in self._w2v.conceptEmbeddings and candidate2 not in self._w2v.conceptEmbeddings:
            return None
        if 'right_context' not in wikilink and 'left_context' not in wikilink:
            return None

        if candidate1 not in self._w2v.conceptEmbeddings:
            return candidate2
        if candidate2 not in self._w2v.conceptEmbeddings:
            return candidate1

        candidate1_vec = self._w2v.conceptEmbeddings[candidate1.lower()]
        candidate2_vec = self._w2v.conceptEmbeddings[candidate2.lower()]

        # tokenize context and project to embedding
        context_vec = self._w2v.meanOfWordList(wikilink['right_context'] + wikilink['left_context'])

        if self._w2v.distance(context_vec, candidate1_vec) < self._w2v.distance(context_vec, candidate2_vec):
            return candidate1
        else:
            return candidate2

if __name__ == "__main__":
    iter_train = WikilinksNewIterator("..\\..\\data\\wikilinks\\train")
    train_stats = WikilinksStatistics(iter_train, load_from_file_path="..\\..\\data\\wikilinks\\train_stats")

    wD = train_stats.mentionLinks
    cD = train_stats.conceptCounts

    print 'Load embeddings...'
    w2v = Word2vecLoader(wordsFilePath="..\\..\\data\\word2vec\\dim300vecs",
                         conceptsFilePath="..\\..\\data\\word2vec\\dim300context_vecs")
    w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)

    print ' ** wordEmbedding size is ',len(w2v.wordEmbeddings)
    print ' ** conceptEmbeddings size is ',len(w2v.conceptEmbeddings)

    context = ["king"]
    context_vec = w2v.meanOfWordList(context)
    print context_vec[0:10]
    context = ["castle"]
    context_vec = w2v.meanOfWordList(context)
    print context_vec[0:10]
    context = ["king", "castle"]
    context_vec = w2v.meanOfWordList(context)
    print context_vec[0:10]
