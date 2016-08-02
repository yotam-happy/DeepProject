import mysql.connector
import re

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
        if cache:
            self.cachePageInfoTable()

    def getConceptTitle(self, conceptId):
        """
        Gets the wikipedia page title for a given id.
        """
        query = "SELECT title FROM article WHERE id = %s "
        self._cursor.execute(query, (conceptId,))
        res = self._cursor.fetchone()
        return res[0].decode("utf-8")  if res is not None else None

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
                print "done ", i
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
        i = 0
        page_id, namespace, title, redirect = self.getPageInfoByTitle(title)
        while page_id is not None and \
                redirect > -1 and \
                i < 3:
            page_id, namespace, title, redirect = self.getPageInfoById(redirect)
            i+=1
        return page_id




if __name__ == "__main__":
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    print wikiDB.getConceptTitle(147676)
    print wikiDB.getPageInfoById(147676)
    print wikiDB.resolvePage('jaguar_cars')
