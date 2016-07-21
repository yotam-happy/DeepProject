import mysql.connector
import re

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
        self._cursor = self._cnx.cursor(buffered=True)

    def getConceptTitle(self, conceptId):
        """
        Gets the wikipedia page title for a given id.
        """
        query = "SELECT title FROM article WHERE id = %s "
        self._cursor.execute(query, (conceptId,))
        return self._cursor.fetchone()[0].decode("utf-8")

    def getConceptTitle(self, conceptId):
        """
        Gets the wikipedia page title for a given id.
        """
        query = "SELECT title FROM article WHERE id = %s "
        self._cursor.execute(query, (conceptId,))
        return self._cursor.fetchone()[0].decode("utf-8")

    def getPageInfoByTitle(self, title):
        query = "SELECT page_id, namespace, title, redirect FROM pages_redirects WHERE title = %s"
        self._cursor.execute(query, (self.stripTitle(title),))
        row = self._cursor.fetchone()
        if row == None:
            return (None,None,None,None)
        return (row[0], row[1], row[2].decode("utf-8"), row[3])

    def getPageInfoById(self, page_id):
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
            page_id, namespace, title, redirect = self.getPageInfoById(page_id)
            i+=1
        return page_id




if __name__ == "__main__":
    wikiDB = WikipediaDbWrapper(user='root', password='???', database='wikiprep-esa-en20151002')
    print wikiDB.getConceptTitle(264210 )
    print wikiDB.getConceptTitle(264209)
