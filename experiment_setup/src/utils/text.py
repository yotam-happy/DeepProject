import unicodedata
import re


def normalize_unicode(str):
    s = str.decode("utf-8", 'ignore').lower()
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore')
    return s


def strip_wiki_title(title):
    t = re.sub('[^0-9a-zA-Z]+', '_', normalize_unicode(title))
    if len(t) > 1 and t[0] == '_':
        t = t[1:]
    if len(t) > 1 and t[-1] == '_':
        t = t[:-1]
    return t
