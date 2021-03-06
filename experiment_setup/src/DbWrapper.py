import mysql.connector
import re
import urllib
import utils.text
class WikipediaDbWrapper:
    """
    Some initial efforts at supporting Db for the project
    Note this class is not thread safe or anything so be aware...
    """

    def __init__(self, user, password, database, host='127.0.0.1', concept_filter=None):
        """
        All the parameters are self explanatory...
        """
        self._user = user
        self._password = password
        self._database = database
        self._host = host

        self._cnx = mysql.connector.connect(user=user, password=password, host=host, database=database,
                                            connection_timeout=1)
        self._cursor = self._cnx.cursor(buffered=True)
        self._articleInlinks = None

        self.concept_filter = concept_filter

        self._cache_resolve = dict()
        self._cache_title = dict()

    def resetConnection(self):
        try:
            self._cursor.close()
        except:
            print "some error"
        try:
            self._cnx.close()
        except:
            print "some error"
        self._cnx = mysql.connector.connect(user=self._user, password=self._password, host=self._host,
                                            database=self._database, connection_timeout=1)
        self._cursor = self._cnx.cursor(buffered=True)

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

    def getPageTitle(self, page_id):
        if page_id in self._cache_title:
            return self._cache_title[page_id]

        query = "SELECT page_title FROM page " \
                "WHERE page_id = %s and page_namespace = 0"
        self._cursor.execute(query, (page_id,))
        row = self._cursor.fetchone()
        ret = row[0] if row is not None else None
        self._cache_title[page_id] = ret
        return ret

    def resolvePage(self, title, verbose=False, print_errors=False, use_pagelink_table=False):
        if title in self._cache_resolve:
            return self._cache_resolve[title]

        for i in xrange(3):
            try:
                ret = self._resolvePage(title,
                                         verbose=verbose,
                                         print_errors=print_errors,
                                         use_pagelink_table=use_pagelink_table)
                self._cache_resolve[title] = ret
                return ret

            except:
                print "reseting connection..."
                self.resetConnection()
        if verbose or print_errors:
            print "could not resolve due to connection problems"
        return None

    def _resolvePage(self, title, verbose=False, print_errors=True, use_pagelink_table=False):
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
            if verbose or print_errors:
                print "could not find page (in main namespace): ", title
            return None

        page_id = int(row[0])
        page_red = int(row[1])
        page_title = row[2]
        if verbose:
            print "got page id =", page_id, "; title =", page_title, "; redirect =", page_red

        c = 0
        while page_red == 1:
            if c == 5:
                if verbose or print_errors:
                    print "too many redirects"
                return None
            c += 1

            # query next page using redirect table
            query = "SELECT page_id, page_is_redirect, page_title FROM page " \
                    "WHERE page_namespace = 0 AND page_title IN " \
                    "(SELECT rd_title FROM redirect WHERE rd_namespace = 0 AND rd_from = %s)"
            self._cursor.execute(query, (page_id,))
            row = self._cursor.fetchone()

            if row is None and use_pagelink_table:
                if verbose:
                    print "redirect not found in redirect table, trying pagelink..."
                # try using pagelink (some older redirects can only be found here)
                query = "SELECT page_id, page_is_redirect, page_title FROM page " \
                        "WHERE page_namespace = 0 AND page_title IN " \
                        "(SELECT pl_title FROM pagelink WHERE pl_namespace = 0 AND pl_from = %s)"
                self._cursor.execute(query, (page_id,))
                row = self._cursor.fetchone()

            if row is None:
                if verbose or print_errors:
                    print "could not resolve redirect for", page_id
                return None

            page_id = int(row[0])
            page_red = int(row[1])
            page_title = row[2]
            if verbose:
                print "got page id =", page_id, "; title =", page_title, "; redirect =", page_red

        if self.concept_filter is not None and page_id not in self.concept_filter:
            if verbose:
                print "concept", page_id, " is filtered"
            return None

        if verbose:
            print "return", page_id
        return page_id

#if __name__ == "__main__":
#    wikiDB = WikipediaDbWrapper(user='yotam', password='rockon123', database='wiki20151002')
#    wikiDB.updatePageTableTitleForLookupColumn()
