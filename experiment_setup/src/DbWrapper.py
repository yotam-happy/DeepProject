import mysql.connector
import re
import urllib
import utils.text
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
        self._articleInlinks = None

        self.cc = 0
        self.cc1 = 0
        self.cc2 = 0
        self.cc3 = 0

    def getArticleTitleById(self, conceptId):
        """
        Gets the wikipedia page title for a given id.
        """
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
        query = "SELECT id FROM article where title_resolver=%s"
        self._cursor.execute(query, (title,))
        row = self._cursor.fetchone()
        if row == None:
            return None
        return int(row[0])

    def updatePageTableTitleForLookupColumn(self):
        self._cnx.autocommit = False
        query = "SELECT page_title, page_id FROM page"
        i = 0
        fetch_cursor = self._cnx.cursor(buffered=True)
        fetch_cursor.execute(query)
        print 'updating'
        while True:
            row = fetch_cursor.fetchone()
            if not row:
                break
            t = utils.text.strip_wiki_title(row[0])
            self._cursor.execute("""
            UPDATE page SET page_title_for_lookup=%s WHERE page_id=%s
            """, (t, int(row[1])))

            i += 1
            if i % 10000 == 0:
                print "updated ", i
            if i % 1000000 == 0:
                self._cnx.commit()


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

    def getPageInfoByTitle(self, title):
        query = "SELECT page_id, namespace, title, redirect FROM pages_redirects WHERE title = %s"
        self._cursor.execute(query, (utils.text.strip_wiki_title(title),))
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

    def resolvePage2(self, title, verbose=False, use_pagelink_table=False):
        '''
        Resolving a page id.
        We first use utils.text.strip_wiki_title to compute a cleaned title
        We then query the page table.
        If this is indicated as a redirect page then we query the redirect table
        If the redirect does not appear in the redirect table then we try the pagelink table as older
        redirects might be found there.
        We take into account that a chain of redirects might be encountered
        If the page at the end of the chain is still redirect we return None
        :param title:               title of page
        :param verbose:             true to show the steps along the way
        :param use_pagelink_table: if true we try the pagelink table as described above
        :return:                    id of page or None if we couldn't resolve
        '''
        if verbose:
            print "resolving title: ", title

        # sometimes the titles come with url type quating (e.g 'the%20offspring')
        title = urllib.unquote(title)
        title = utils.text.strip_wiki_title(title)
        if verbose:
            print "lookup key: ", title

        # get page
        query = "SELECT page_id, page_is_redirect, page_title FROM page " \
                "WHERE page_title_for_lookup = %s and page_namespace = 0"
        self._cursor.execute(query, (title,))
        row = self._cursor.fetchone()
        if row is None:
            if verbose:
                print "could not find page (in main namespace)"
            return None

        page_id = int(row[0])
        page_red = int(row[1])
        page_title = row[2]
        if verbose:
            print "got page id =", page_id, "; title =", page_title, "; redirect =", page_red

        while page_red == 1:
            # query next page using redirect table
            query = "SELECT page_id, page_is_redirect, page_title FROM page " \
                    "WHERE page_namespace = 0 AND page_title IN " \
                    "(SELECT rd_title FROM redirect WHERE rd_namespace = 0 AND rd_from = $s)"
            self._cursor.execute(query, (page_id,))
            row = self._cursor.fetchone()

            if row is None and use_pagelink_table:
                if verbose:
                    print "redirect not found in redirect table, trying pagelink..."
                # try using pagelink (some older redirects can only be found here)
                query = "SELECT page_id, page_is_redirect, page_title FROM page " \
                        "WHERE page_namespace = 0 AND page_title IN " \
                        "(SELECT pl_title FROM pagelink WHERE pl_namespace = 0 AND pl_from = $s)"
                self._cursor.execute(query, (page_id,))
                row = self._cursor.fetchone()

            if row is None:
                if verbose:
                    print "could not resolve redirect"
                return None

            page_id = int(row[0])
            page_red = int(row[1])
            page_title = row[2]
            if verbose:
                print "got page id =", page_id, "; title =", page_title, "; redirect =", page_red

        if verbose:
            print "return", page_id
        return page_id

    def resolvePage(self, title, verbose=False):
        '''
        Not trivial.
        '''
        if verbose:
            print "resolving title: ", title
        title = urllib.unquote(title)
        if verbose:
            print "unquoted: ", title

        candidate = self.getArticleIdByTitle(title)
        if verbose:
            print "self.getArticleIdByTitle: ", candidate
        page_id, namespace, title, redirect = self.getPageInfoByTitle(title)
        if verbose:
            print "self.getPageInfoByTitle: id: ", page_id, " redirect: ", redirect
        if self.getArticleTitleById(page_id) is not None:
            if verbose:
                print "self.getArticleTitleById: ", self.getArticleTitleById(page_id)
            candidate = page_id

        i = 0
        if verbose:
            print "resolving redirects"
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

if __name__ == "__main__":
    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
    wikiDB.updatePageTableTitleForLookupColumn()
