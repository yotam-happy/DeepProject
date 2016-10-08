import unicodedata
def normalize_unicode(str):
    print str
    s = str.decode("utf-8", 'ignore').lower()
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore')
    return s
