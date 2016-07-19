import mysql.connector

class WikipediaDbWrapper:
    """
    Some initial efforts at supporting Db for the project
    Note this class is not thread safe or anything so be aware...
    """

    def __init__(self, user, password, database, host='127.0.0.1'):
        """
        All the parameters are self explanatory...
        """

        self._cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self._cursor = self._cnx.cursor()

    def getConceptTitle(self, conceptId):
        """
        Gets the wikipedia page title for a given id.
        """
        query = "SELECT title FROM article WHERE id = %s "
        self._cursor.execute(query, (conceptId,))
        return self._cursor.fetchone()[0].decode("utf-8")

    def getPages(self):
        query = "SELECT id, title, namespace, isRedirect FROM article"
        self._cursor.execute(query)
        for row in self._cursor:
            p = Page(row[0],row[1],row[2],row[3])
            yield p

    def getRedirects(self):
        query = "SELECT from, to, namespace FROM redirects"
        self._cursor.execute(query)
        for row in self._cursor:
            yield (row[0],row[1],row[2])



class Page:
    def __init__(self, id, title, namespace, isRedirect):
        self.id = id
        self.title = title
        self.isRedirect = isRedirect
        self.namespace = namespace
        self.redirectTo = -1

class titleResolver:
    def __init__(self, db):
        self._pagesById = dict()
        self._pagesByTitle = dict()
        for page in db.getPages():
            # we care only about main namespace
            if page.namespace != 0:
                continue;
            self._pagesById[page.id] = page
            self._pagesByTitle[page.title] = page

        for red_from, red_to, red_namespace in db.getRedirects:
            if red_namespace != 0:
                continue
            f = self._pagesById(red_from)
            t = self._pagesByTitle(red_to)
            if f is None or t is None:
                continue
            f.redirectTo = t.id

    def resolvePage(self, title):
        i = 0
        while title in self._pagesByTitle and \
                self._pagesByTitle[title].isRedirect and \
                self._pagesByTitle[title].redirectTo >= 0 and\
                i < 3:
            title = self._pagesByTitle[title].title
            i+=1
        return self._pagesByTitle[title].id if title in self._pagesByTitle else None



if __name__ == "__main__":
    wikiDB = WikipediaDbWrapper(user='root', password='???', database='wikiprep-esa-en20151002')
    print wikiDB.getConceptTitle(264210 )
    print wikiDB.getConceptTitle(264209)
