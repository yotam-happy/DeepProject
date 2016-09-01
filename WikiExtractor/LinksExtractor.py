import os
import nltk
from nltk import TreebankWordTokenizer
from nltk import PunktSentenceTokenizer
import re
import json
import mysql.connector
import cProfile

class WikipediaDbWrapper:
    """
    Some initial efforts at supporting Db for the project
    Note this class is not thread safe or anything so be aware...
    """

    def __init__(self, user, password, database, host='127.0.0.1', cache=False):
        """
        All the parameters are self explanatory...
        """

        self._cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self._cursor = self._cnx.cursor(buffered=True)
        self._pageInfoByIdCache = None
        self._pageInfoByTitleCache = None
        self._articleId = None
        self._articleTitle = None
        if cache:
            self.cachePageInfoTable()
            self.cacheArticleTable()
        self._articleInlinks = None

        self.cc = 0
        self.cc1 = 0
        self.cc2 = 0
        self.cc3 = 0

    def getArticleTitleById(self, conceptId):
        """
        Gets the wikipedia page title for a given id.
        """
        if self._articleId is None:
            raise "Must cache table!"
        return self._articleId[conceptId] if conceptId in self._articleId else None

    def getArticleIdByTitle(self, title):
        """
        Gets the wikipedia page title for a given id.
        """
        if self._articleTitle is None:
            raise "Must cache table!"
        title = self.stripTitle(title)
        return self._articleTitle[title] if title in self._articleTitle else None

    def printSomeArticle(self):
        query = "SELECT title, id FROM article"
        self._cursor.execute(query)
        i = 0
        while True:
            row = self._cursor.fetchone()
            if not row:
                break
            t = row[0].decode("utf-8")
            print row[1], ": ", self.stripTitle(t)
            if i > 100:
                return
            i += 1

    def cacheArticleTable(self):
        query = "SELECT title, id FROM article"
        self._articleId = dict()
        self._articleTitle = dict()

        i = 0
        self._cursor.execute(query)
        print "caching"
        while True:
            row = self._cursor.fetchone()
            if not row:
                break
            t = self.stripTitle(row[0].decode("utf-8"))
            self._articleId[row[1]] = t
            self._articleTitle[t] = row[1]
            if i % 100000 == 0:
                print "caching article ", i
            i += 1
    def cacheArticleInlinksTable(self):
        query = "SELECT target_id, inlink FROM inlinks"
        self._articleInlinks = dict()

        i = 0
        self._cursor.execute(query)
        print "caching"
        while True:
            row = self._cursor.fetchone()
            if not row:
                break
            self._articleInlinks[row[0]] = row[1]
            if i % 100000 == 0:
                print "caching article ", i
            i += 1

    def getInlinks(self, wikiId):
        if self._articleInlinks is None:
            raise "Must cache table!"
        return self._articleInlinks[wikiId] if wikiId in self._articleInlinks else None

    def cachePageInfoTable(self):
        query = "SELECT page_id, namespace, title, redirect FROM pages_redirects where namespace=0"
        self._pageInfoByTitleCache = dict()
        self._pageInfoByIdCache = dict()
        self._cursor.execute(query)
        i = 0
        while True:
            row = self._cursor.fetchone()
            if not row:
                break
            t = row[2].decode("utf-8")
            self._pageInfoByTitleCache[t] = (row[0], row[1], t, row[3])
            self._pageInfoByIdCache[row[0]] = (row[0], row[1], t, row[3])
            if i % 100000 == 0:
                print "caching redirect ", i
            i += 1
        print "cached ", len(self._pageInfoByTitleCache), " entries"

    def getPageInfoByTitle(self, title):
        if self._pageInfoByTitleCache is not None:
            return self._pageInfoByTitleCache[self.stripTitle(title)] \
                if self.stripTitle(title) in self._pageInfoByTitleCache \
                else (None,None,None,None)
        query = "SELECT page_id, namespace, title, redirect FROM pages_redirects WHERE title = %s"
        self._cursor.execute(query, (self.stripTitle(title),))
        row = self._cursor.fetchone()
        if row == None:
            return (None,None,None,None)
        return (row[0], row[1], row[2].decode("utf-8"), row[3])

    def getPageInfoById(self, page_id):
        if self._pageInfoByIdCache is not None:
            return self._pageInfoByIdCache[page_id] \
                if page_id in self._pageInfoByIdCache \
                else (None,None,None,None)
        query = "SELECT page_id, namespace, title, redirect FROM pages_redirects WHERE page_id = %s"
        self._cursor.execute(query, (page_id,))
        row = self._cursor.fetchone()
        if row == None:
            return (None,None,None,None)
        return (row[0], row[1], row[2].decode("utf-8"), row[3])

    def stripTitle(self,title):
        t = re.sub('[^0-9a-zA-Z]', '_', title.lower())
        return t

    def resolvePage(self, title):
        '''
        Not trivial.
        '''
        title = urllib.unquote(title)
        candidate = self.getArticleIdByTitle(title)
        page_id, namespace, title, redirect = self.getPageInfoByTitle(title)
        if self.getArticleTitleById(page_id) is not None:
            candidate = page_id

        i = 0
        while page_id is not None and \
                redirect > -1 and \
                i < 3:
            page_id, namespace, title, redirect = self.getPageInfoById(redirect)
            if page_id is not None and self.getArticleTitleById(page_id) is not None:
                candidate = page_id
            elif title is not None:
                x = self.getArticleIdByTitle(title)
                if x is not None:
                    candidate = x
            i+=1
        return candidate

def isGoodToken(token):
    if len(token.strip()) == 0:
        return False
    return True

def fixURL(url):
    url = re.sub("%[0-9a-fA-F][0-9a-fA-F]","_", url)
    return url

word_tok = TreebankWordTokenizer()
sent_tok = PunktSentenceTokenizer()

def word_tokenize(text, language='english'):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    return [token for sent in sent_tok.tokenize(text, language)
            for token in word_tok.tokenize(sent)]

def docToLinks(doc, db, contextLen = 25):

    # record mentions
    links = re.findall('<a href="?\'?([^"\'>]*)">([^<]*)</a>', doc)

    line = re.sub(r'</a>', ' yotlinkendyot ', doc)
    line = re.sub(r'<a href="?\'?([^"\'>]*)">', " yotlinkstartyot ", line)
    tokens = word_tokenize(line.decode('utf-8'))
    tokens = [token.strip() for token in tokens if isGoodToken(token)]

    linkBounds = []
    only_words = []
    insideLink = False
    i = 0
    for token in tokens:
        if token == "yotlinkendyot":
            if insideLink:
                insideLink = False
                rightContext = i
                linkBounds.append((leftContext, rightContext))
        elif token == "yotlinkstartyot" and not insideLink:
            # if we are already inside a link then something is wrong. just skip
            if insideLink:
                insideLink = False
            else:
                insideLink = True
                leftContext = i
        else:
            only_words.append(token)
            i += 1

    for link, bounds in zip(links, linkBounds):
        l = bounds[0] - contextLen if bounds[0] - contextLen >= 0 else 0
        r = bounds[1] + contextLen if bounds[1] + contextLen < len(only_words) else len(only_words)

        leftContext = only_words[l:bounds[0]]
        rightContext = only_words[bounds[1]:r]
        link_url, mention = link
        link_url_fixed = fixURL(link_url)
        link_id = db.resolvePage(link_url_fixed)
        yield {'word': mention,
               'wikiId': link_id,
               'left_context': leftContext,
               'right_context': rightContext,
               'wikiurl':link_url}

def textDirectoryIter(path):
    '''
    Iterates all files in directory as text files. Returns one line at a time
    This is ok since wikipedia places each paragraph in a single line so we have
    proper context for links in a paragraph
    :param path: directory path
    '''
    if not os.path.isdir(path):
        raise "Path is not a directory: " + path
    for file in os.listdir(path):
        print "Opening file ", os.path.join(path,file)
        lines = open(os.path.join(path,file), "r").readlines()
        doc = []
        for line in lines:
            doc.append(line)
            if (line.startswith('</doc>')):
                yield '\n'.join(doc)
                doc = []

def wikiToLinks(wiki_dir, out_dir, db, context_len = 25, jsons_per_file = 400000):
    c = 0
    outfile_c = 0
    outlink_c = 0
    out_lines = []
    for i, doc in enumerate(textDirectoryIter(wiki_dir)):
        for link in docToLinks(doc, db, contextLen=context_len):
            if link['wikiId'] == None:
                # skip if we cant resolve id
                continue
            out_lines.append(json.dumps(link) + "\n")
            outlink_c += 1
            c += 1
            if len(out_lines) >= jsons_per_file:
                fname = os.path.join(out_dir, "wikilinks_" + str(outfile_c) + ".json")
                out_f = open(fname, "w")
                out_f.writelines(out_lines)
                out_f.close()
                outlink_c = 0
                outfile_c += 1
                out_lines = []
                print "written file: ", fname, " (", i, " documents and ", c, " links)"
    if len(out_lines) >= 0:
        out_f = open(os.path.join(out_dir, "wikilinks_" + str(outfile_c) + ".json"), "w")
        out_f.writelines(out_lines)
        out_f.close()

if __name__ == '__main__':

    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002', cache=True)
    wikiToLinks('/home/yotam/pythonWorkspace/deepProject/data/wikiextractor-output',
                '/home/yotam/pythonWorkspace/deepProject/data/wikipedia-intralinks', wikiDB)
    print "done"
