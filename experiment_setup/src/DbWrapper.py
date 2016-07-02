import mysql.connector

class WikipediaDbWrapper:
    def __init__(self, user, password, database, host='127.0.0.1'):
        self._cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self._cursor = self._cnx.cursor()

    def getConceptTitle(self, conceptId):
        query = "SELECT title FROM article WHERE id = %s "
        self._cursor.execute(query, (conceptId,))
        return self._cursor.fetchone()[0].decode("utf-8")

if __name__ == "__main__":
    wikiDB = WikipediaDbWrapper(user='root', password='???', database='wikiprep-esa-en20151002')
    print wikiDB.getConceptTitle(264210 )
    print wikiDB.getConceptTitle(264209)
