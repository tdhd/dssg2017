'''
the line end should either be \r or \n
'''
RISSTART = '\d+\. \r'

def read_article(lines):
    article = {}
    for line in lines:
        keyValue = line.split("-")
        if "-" in line and len(keyValue) == 2:
            key,value = [t.strip() for t in keyValue]
            if key in article: article[key] += "," + value
            else: article[key] = value
    return article

def read_ris_lines(lines):
    import pandas as pd
    import re
    startArticle = [idx for idx,l in enumerate(lines) if re.match(RISSTART, l)]
    articles = [read_article(lines[startArticle[s]:startArticle[s+1]]) for s in range(len(startArticle)-1)]
    return pd.DataFrame(articles)

def read_ris(fn):
    lines = open(fn,"rt").readlines()
    return read_ris_lines(lines)
