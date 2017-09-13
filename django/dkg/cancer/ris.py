import re

RISSTART = '\r'

IGNORE_KW = [x.strip() for x in open("ignore_keywords.txt").readlines()]

def contains_ignore_words(line):
    return any([iw in line.lower() for iw in IGNORE_KW])

def read_article(lines):
    '''
    Reads single RIS tag
    '''
    article = {}
    for line in lines:
        match = re.match("[A-Z0-9]{2}\s+-",line)
        if match:
            key,value = line[:2], line[match.span()[1]:].strip()
            # Filtering out lines that contain to-be-ignored keywords
            if not contains_ignore_words(line):
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
        # RIS entry starts with journal tag 
        # (https://de.wikipedia.org/wiki/RIS_(Dateiformat))
        if k == 'TY': 
            ris.insert(0,"TY  - " + v)
        elif k == 'KW':
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
    # Double check and insert 'start'/journal tag
    if "TY" not in ris[0]:
        ris.insert(0,"TY  - Missing Journal")
    # add closing tag
    ris.append("ER  -")
    return "\n".join(ris)


def write_ris_lines(fn, df):
    """
    Writes a pandas dataframe to RIS file.
    """
    with open(fn, 'w') as fh:
        for _, article in df.iterrows():
            fh.write(write_article(article) + "\n\n")
