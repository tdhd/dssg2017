import re

RISSTART = '\r'

def read_article(lines):
    article = {}
    for line in lines:
        match = re.match("[A-Z0-9]{2}\s+-",line)
        if match:
            key,value = line[:2], line[match.span()[1]:].strip()
            if key in article: article[key] += "," + value
            else: article[key] = value
    return article

def read_ris_lines(lines):
    import pandas as pd
    startArticle = [idx for idx,l in enumerate(lines) if re.match(RISSTART, l)]
    articles = [read_article(lines[startArticle[s]:startArticle[s+1]]) for s in range(len(startArticle)-1)]
    return pd.DataFrame(articles)
