import mysql.connector
import nltk
import re
import mysql
from nltk.corpus import stopwords

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


stop = stopwords.words('english')

def isGoodToken(token):
    if re.search('[a-zA-Z]', token) == None:
        return False
    return True

def fixURL(url):
    url = re.sub("%20"," ", url)
    return url

def docToLinks(doc, db, contextLen = 25):

    lines = doc.split('\n')
    for line in lines:
        # record mentions
        links = re.findall('<a href="?\'?([^"\'>]*)">([^<]*)</a>', line)

        line = re.sub(r'</a>', ' _linkend_ ', line)
        line = re.sub(r'<a href="?\'?([^"\'>]*)">', " _linkstart_ ", line)
        tokens = nltk.word_tokenize(line)
        tokens = [token for token in tokens if isGoodToken(token)]

        linkBounds = []
        only_words = []
        insideLink = False
        i = 0
        for token in tokens:
            if token == "_linkend_":
                # sanity check
                if not insideLink:
                    raise Exception("Something is wrong")
                insideLink = False

                rightContext = i
                linkBounds.append((leftContext, rightContext))
            elif token.startswith('_link'):
                # sanity check
                if insideLink:
                    raise Exception("Something is wrong")
                insideLink = True

                leftContext = i
            else:
                only_words.append(token)
                i += 1

        print only_words
        for link, bounds in zip(links, linkBounds):
            l = bounds[0] - contextLen if bounds[0] - contextLen >= 0 else 0
            r = bounds[1] + contextLen if bounds[1] + contextLen < len(only_words) else len(only_words)
            leftContext = only_words[l:bounds[0]]
            rightContext = only_words[bounds[1]:r]
            link_url, mention = link
            link_url = fixURL(link_url)
            link_id = db.resolvePage(link_url)
            print link_url, " (", link_id, ") as ", mention
            print bounds
            print leftContext
            print rightContext
            if link_id == None:
                raise "Whaaaat!"


if __name__ == '__main__':

    wikiDB = WikipediaDbWrapper(user='root', password='rockon123', database='wikiprep-esa-en20151002')
    docToLinks('The Angolan Armed Forces (<a href="Portuguese%20language">Portuguese</a>: "Forças Armadas Angolanas") are the <a href="military">military</a> in <a href="Angola">Angola</a> that succeeded the <a href="Armed%20Forces%20for%20the%20Liberation%20of%20Angola">Armed Forces for the Liberation of Angola</a> (FAPLA) following the abortive <a href="Bicesse%20Accord">Bicesse Accord</a> with the National Union for the Total Independence of Angola (<a href="UNITA">UNITA</a>) in 1991. As part of the peace agreement, troops from both armies were to be <a href="demilitarized">demilitarized</a> and then integrated. Integration was never completed as UNITA went back to war in 1992. Later, consequences for UNITA members in Luanda were harsh with FAPLA veterans persecuting their erstwhile opponents in certain areas and reports of <a href="vigilantism">vigilantism</a>.', wikiDB)
