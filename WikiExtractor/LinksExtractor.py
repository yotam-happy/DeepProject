import nltk
import re
import mysql
from nltk.corpus import stopwords

class docTitles:
    def __init__(self, user, password, database, host='127.0.0.1'):
        self.titleToId = dict()
        self._cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self._cursor = self._cnx.cursor()
        self.getAllTitles()

    def getAllTitles(self):
        self._cursor.execute("SELECT id, title FROM article")
        for row in self._cursor:
            self.titleToId[row[1].decode("utf-8")] = row[0]
        print self.titleToId

def subF(matchobj):
    link = matchobj.group(1)
    link = re.sub(r'%20', '_', link)
    return " _link_" + link + "_ "

stop = stopwords.words('english')

def isGoodToken(token):
    if re.search('[a-zA-Z]', token) == None:
        return False
    return True


def docToLinks(doc, contextLen = 25):

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
            print link
            print bounds
            print leftContext
            print rightContext


if __name__ == '__main__':

    docToLinks('The Angolan Armed Forces (<a href="Portuguese%20language">Portuguese</a>: "Forças Armadas Angolanas") are the <a href="military">military</a> in <a href="Angola">Angola</a> that succeeded the <a href="Armed%20Forces%20for%20the%20Liberation%20of%20Angola">Armed Forces for the Liberation of Angola</a> (FAPLA) following the abortive <a href="Bicesse%20Accord">Bicesse Accord</a> with the National Union for the Total Independence of Angola (<a href="UNITA">UNITA</a>) in 1991. As part of the peace agreement, troops from both armies were to be <a href="demilitarized">demilitarized</a> and then integrated. Integration was never completed as UNITA went back to war in 1992. Later, consequences for UNITA members in Luanda were harsh with FAPLA veterans persecuting their erstwhile opponents in certain areas and reports of <a href="vigilantism">vigilantism</a>.')
