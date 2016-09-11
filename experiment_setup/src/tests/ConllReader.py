def _CoNLLFileToDocIterator(fname, split='testb'):
    f = open(fname,'r')
    lines = f.readlines()

    curdocName = None
    curdocSplit = None
    curdoc = None

    for line in lines:
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            if curdocName is not None and (split is None or curdocSplit == split):
                yield (curdoc, curdocName)
            sp = line.split(' ')
            curdocName = sp[2][:-1]
            curdocSplit = 'testa' if sp[1].endswith('testa') else ('testb' if sp[1].endswith('testb') else 'train')
            curdoc = []
        else:
            curdoc.append(line)
    if curdocName is not None and (split is None or curdocSplit == split):
        yield (curdoc, curdocName)


def _CoNLLDocToSentenceIterator(docLines):
    sent = [line for line in docLines if line]
    yield sent
#    sent = []
#    for line in docLines:
#        if not line:
#            if len(sent) > 0:
#                yield sent
#            sent = []
#        else:
#            sent.append(line)

def _CoNLLRawToTuplesIterator(lines):
    '''
    yields tuples:
    (surface,ismention,islinked,YAGO2,WikiURL,WikiId,FB)
    surface is eaither a word or the full mention
    '''
    for line in lines:
        if len(line) == 0:
            # sentence boundary.
            continue
        t = line.split('\t')
        if len(t) == 1:
            yield (t[0], False, None, None, None, None, None)
        else:
            if t[1] != 'B':
                continue
            if t[3] == '--NME--':
                yield (t[2], True, False, None, None, None, None)
            else:
                yield (t[2], True, True, t[3], t[4], t[5], t[6] if len(t) >= 7 else None)

def CoNLLWikilinkIterator(fname, split='testb', includeUnresolved=False):
    for (doc,docName) in _CoNLLFileToDocIterator(fname, split):
        for sent in _CoNLLDocToSentenceIterator(doc):
            sent = [token for token in _CoNLLRawToTuplesIterator(sent)]
            for i, token in enumerate(sent):
                if token[1] and (includeUnresolved or token[2]):
                    wlink = dict()
                    wlink['wikiurl'] = token[4]
                    wlink['yago2'] = token[3]
                    wlink['wikiId'] = token[5] # BE CAREFUL!! This might be a different mapping then ours
                    wlink['word'] = token[0]

                    left_context_text = ' '.join([x[0] for x in sent[:i+1]])
                    right_context_text = ' '.join([x[0] for x in sent[i:]])
                    wlink['left_context_text'] = left_context_text
                    wlink['right_context_text'] = right_context_text
                    wlink['left_context'] = left_context_text.split(' ')
                    wlink['right_context'] = right_context_text.split(' ')

                    yield wlink

def CoNLLasDocs(fname, split='testb'):
    for (doc,docName) in _CoNLLFileToDocIterator('../data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv', 'testb'):
        docWords = []
        for sent in _CoNLLDocToSentenceIterator(doc):
            docWords += [token[0] for token in _CoNLLRawToTuplesIterator(sent)]
        print docName, ": ", docWords
        yield docName, docWords
