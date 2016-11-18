import nltk
from entity_vector import EntityVector
import numpy as np

class YamadaEmbedder:
    def __init__(self, path, db=None):
        self.entvec = EntityVector.load(path)
        self._cache = dict()
        self._db = db
        return

    def similarity(self, v1, v2):
        return np.dot(v1, v2)

    def text_to_embedding(self, text, mention):
        tagged = nltk.pos_tag(text)
        vecs = []
        nn = 0
        for word, tag in tagged:
            if tag.startswith('N') and not mention.lower().find(word.lower()) > -1:
                try:
                    w = self.entvec.get_word_vector(unicode(word.lower()))
                    if w is not None:
                        vecs.append(w)
                        nn += 1
                except:
                    pass
        vecs = np.array(vecs)
        return vecs.mean(0)

    def entity_embd(self, title):
        try:
            return self.entvec.get_entity_vector(title)
        except:
            return None

    def from_the_cache(self, page_id):
        return self._cache[page_id] if page_id in self._cache else None

    def into_the_cache(self, page_id, url, title, verbose=False):
        if page_id in self._cache:
            return self._cache[page_id]

        try:
            if title is not None:
                embd = self.entity_embd(title)
                if embd is not None:
                    self._cache[page_id] = embd
                    return embd

            title_from_url = unicode(url.decode("utf-8"))
            title_from_url = title_from_url[title_from_url.rfind('/') + 1:]
            title_from_url = title_from_url.replace('_', ' ')
            embd = self.entity_embd(title_from_url)
            if embd is not None:
                self._cache[page_id] = embd
                return embd

            url_by_id = self._db.getPageTitle(page_id)
            url_by_id = unicode(url_by_id.decode("utf-8"))
            url_by_id = url_by_id.replace('_', ' ')
            embd = self.entity_embd(url_by_id)
            if embd is not None:
                self._cache[page_id] = embd
                return embd
        except Exception as e:
            if verbose:
                print url, "some error", e
            return None
        if verbose:
            print url, "not resolved"
        return None


if __name__ == "__main__":
   embedder = YamadaEmbedder('../data/yamada/enwiki_entity_vector_500_20151026.pickle')
   txt = "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep ."
   txt = txt.split(' ')
   print embedder.text_to_embedding(txt)