import re

RISSTART = '\r'


def read_article(lines):
    '''
    Reads single RIS tag
    '''
    article = {}
    for line in lines:
        match = re.match("[A-Z0-9]{2}\s+-",line)
        if match:
            key,value = line[:2], line[match.span()[1]:].strip()
            # keywords to list
            if key in article: article[key] += [value]
            else: article[key] = [value]
    return article


def read_ris_lines(lines):
    '''
    RIS file import
    INPUT:
    lines   lines of RIS file
    OUTPUT:
    pandas dataframe, columns corresponding to RIS tags
    '''

    import pandas as pd
    startArticle = [idx for idx,l in enumerate(lines) if re.match(RISSTART, l)]
    articles = [read_article(lines[startArticle[s]:startArticle[s+1]]) for s in range(len(startArticle)-1)]
    return pd.DataFrame(articles)


def write_article(article):
    """
    Writes single article to RIS string
    the column 'keywords' is assumed to contain the assigned keywords as tuples
    ('label','score','anotator')
    TODO: better keyword-list flattening, currently only labels are extracted
            problem is that there is only the KW RIS tag,
            we'd need annotator/score specific RIS tags
    """
    ris = []
    for k, v in article.iteritems():
        if k == 'KW':
            for kw in v:
                ris.append("KW  - " + kw[0])
        elif k == 'A1':  # primary authors
            for vv in v:
                ris.append("A1  - " + vv)
        elif k == 'A2':  # secondary authors
            for vv in v:
                ris.append("A2  - " + vv)

        else:
            ris.append(str(k) + "  - " + str(v))
    return "\n".join(ris)


def write_ris_lines(fn, df):
    """
    Writes a pandas dataframe to RIS file.
    """
    with open(fn, 'w') as fh:
        for _, article in df.iterrows():
            fh.write(write_article(article) + "\n\n")
