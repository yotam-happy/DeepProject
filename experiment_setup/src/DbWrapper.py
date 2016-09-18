import mysql.connector
import re
import urllib

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
        if self._articleId is not None:
            return self._articleId[conceptId] if conceptId in self._articleId else None
        else:
            query = "SELECT title_resolver FROM article where id=%s"
            self._cursor.execute(query, (conceptId,))
            row = self._cursor.fetchone()
            if row == None:
                return None
            return row[0]

    def getArticleIdByTitle(self, title):
        """
        Gets the wikipedia page title for a given id.
        """
        title = self.stripTitle(title)
        if self._articleTitle is not None:
            return self._articleTitle[title] if title in self._articleTitle else None
        else:
            query = "SELECT id FROM article where title_resolver=%s"
            self._cursor.execute(query, (title,))
            row = self._cursor.fetchone()
            if row == None:
                return None
            return int(row[0])

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

    def updateTables(self):
        print "self._cnx.autocommit:", self._cnx.autocommit
        self._cnx.autocommit = False
        n = 0
        print "upating article table"
        for id, title_processed in self._articleId.items():
            self._cursor.execute("""
            UPDATE article SET title_resolver=%s WHERE id=%s
            """,(title_processed, id))
            n+=1
            print n
            if (n % 10000) == 0:
                print "done ", n

        n = 0
        print "upating pages_redirects table"
        for id, s in self._pageInfoByIdCache.items():
            self._cursor.execute("""
            UPDATE pages_redirects SET title_resolver=%s WHERE page_id=%s
            """, (title_processed, s[2]))
            n += 1
            if (n % 10000) == 0:
                print "done ", n
        self._cnx.commit()

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


    def resolvePage2(self, title):
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
            i += 1
        return candidate


if __name__ == "__main__":
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    wikiDB.cacheArticleTable()
    wikiDB.printSomeArticle()
