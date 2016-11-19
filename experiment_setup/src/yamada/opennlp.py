from jsonrpclib.jsonrpc import ServerProxy


def _parse_tree(parse_str):
    """
    returns a 2-tuple. first item is the tag, second is either a word or a list representing a sub-tree

    :param self:
    :param parse_str:
    :return:
    """
    if parse_str[0] != '(':
        raise "error, parse string malformed!"
    rest = parse_str[1:]

    i = rest.find(' ')
    if i < 1:
        raise "error, parse string malformed!"
    tag = rest[:i]
    rest = rest[i:]

    # if we reached a leaf.
    if rest[1] != '(':
        rest = rest[1:]
        i = rest.find(')')
        k = rest.find('(')
        if k >=0 and k < i:
            raise "error, parse string malformed!"
        node = rest[:i]
        rest = rest[i:]
        return (tag, node), rest

    node_list = []
    while rest.startswith(' ('):
        rest = rest[1:]
        subtree, rest = _parse_tree(rest)
        node_list.append(subtree)
        if rest[0] != ')':
            raise "error, parse string malformed!"
        rest = rest[1:]
    tree = (tag, node_list)
    if len(rest) == 1:
        return tree
    else:
        return tree, rest


def print_tree(tree, sp=''):
    s = sp
    tag, node = tree
    s += tag
    if type(node) is not list:
        s += ' ' + str(node)
        print s
    else:
        print s
        for subree in node:
            print_tree(subree, sp + "  ")


def _list_nouns(tree, l):
    tag, node = tree
    if type(node) is not list:
        if tag.startswith('N'):
            l.append(node)
    else:
        for subree in node:
            _list_nouns(subree, l)


class OpenNLP:
    def __init__(self, host='localhost', port=8080):
        uri = "http://%s:%d" % (host, port)
        self.server = ServerProxy(uri)

    def parse(self, text):
        p = self.server.parse(text)
        try:
            return _parse_tree(p)
        except:
            print "problem with parse tree:", p
            return None

    def list_nouns(self, text):
        l = []
        p = self.parse(text)
        if p is not None:
            _list_nouns(self.parse(text), l)
        return l


if __name__ == '__main__':
    nlp = OpenNLP()
    results = nlp.parse("Shhh  (Be vewy vewy quiet), I'm hunting wabbits .")
    print_tree(results)
    print nlp.list_nouns("Shhh  (Be vewy vewy quiet), I'm hunting wabbits .")