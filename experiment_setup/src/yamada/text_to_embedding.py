import nltk
from entity_vector import EntityVector
import numpy as np

class YamadaEmbedder:
    def __init__(self, path):
        self.entvec = EntityVector.load(path)
        return

    def text_to_embedding(self, text):
        tagged = nltk.pos_tag(text)
        vecs = []
        for word, tag in tagged:
            if tag.startswith('N'):
                print 'trying', word
                w = self.entvec.get_word_vector(unicode(word.lower()))
                if w is not None:
                    vecs.append(w)
        vecs = np.array(vecs)
        return vecs.mean(0)

    def entity_embd(self, title):
        return self.entvec.get_entity_vector(title)

#if __name__ == "__main__":
#   embedder = YamadaEmbedder('../data/yamada/enwiki_entity_vector_500_20151026.pickle')
#   txt = "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep ."
#   txt = txt.split(' ')
#   print embedder.text_to_embedding(txt)