from WikilinksStatistics import *
from Word2vecLoader import *
from nltk import word_tokenize

class EsaSimilarityModel:
    """
    This model is naivly compares word context pairs with cosine similarity
    the context vector is claculated a simple mean on its content in the vector space"""

    def __init__(self, iter=None, stats_file=None):
        """
        :param iter:        iterator for the training set
        :param stats_file:  path of a stats file, instead of using the iterator to calculate
                            statistics (takes some time)
        """
        self._wikilink_stats = WikilinksStatistics(iter, load_from_file_path=stats_file)
        if stats_file is None:
            self._wikilink_stats.calcStatistics()

        self._w2v_loader = Word2vecLoader(wordsFilePath="..\\..\\data\\word2vec\\dim300vecs",
                         conceptsFilePath="..\\..\\data\\word2vec\\dim300context_vecs")

    def predict(self, wikilink):
        """
        perdict for wikilink what sense is closer to ots context according to the its mentionLinks
        :param wikilink:    the wikilink structure
        :return: predicted sense0
        """
        if self._wikilink_stats is None:
            raise Exception("Similarity model must have statistics object")

        if not (wikilink["word"] in self._wikilink_stats.mentionLinks):
            return None

        # get statistics for word
        links = self._wikilink_stats.mentionLinks[wikilink["word"]]

        # projects sense and context to vector space
        # TODO: 1.check if links format is valid or conceptDict
        # TODO: 2. model to turn context word to vec
        self._w2v_loader.loadEmbeddings(wordDict=wikilink["word"], conceptDict=links)
        context = word_tokenize(wikilink['right_context'] + wikilink['left_context'])
        context_vec = [self._w2v_loader.transform(word.lower()) for word in context if word.isalpha()]

        # concept = max(links, key=lambda k: k[1])

        # calculates similarity for each pair in the knockout
        concept = 0
        return concept

if __name__ == '__main__':
    print 'starting main'


"""
# some script for the python console to get wlink structure
"""
from zipfile import ZipFile
import pandas as pd # pandas
zf = ZipFile("C:\Users\Noam\Documents\GitHub\DeepProject\Data\wikilink.zip", 'r') # Read in a list of zipped files
file = zf.open(zf.namelist()[0])
df = pd.read_json(file)
for wlink in df.wlinks:
    if(not 'wikiId' in wlink):
        continue
    break



