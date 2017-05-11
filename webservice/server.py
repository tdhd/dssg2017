import bjoern
import json

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy.sparse import hstack
import warnings,json,gzip,re
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from labels import to_list, flat_map, classifications_list_to_cancer_paper_subs, classifications_list_to_cancer_paper_subs_pruned, paper_types_from, cancer_types_from


from ris import read_ris_lines

def respondWith(filename, response):
    response('200 OK', [('Content-Type', 'text/html')])
    with open(filename, 'r') as f:
        c = f.readlines()
        return ['\n'.join(c)]

def readInput(env):
    return env.get("wsgi.input").getvalue()

def risFrom(inp):
    # skip encoding beging and end
    lines = inp.split("\n")[3:-3]
    df = read_ris_lines(lines)
    return df

def risFeatures(df):
    print("Vectorizing title character ngrams")
    titleVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**15)
    titleVects = titleVectorizer.fit_transform(df.T1.fillna(""))
    # print("Vectorizing keywords")
    # keywordVects = CountVectorizer().fit_transform(df.KW.str.replace('[\[\]\'\"]',""))
    print("Vectorizing authors")
    authorVects = HashingVectorizer(n_features=2**15).fit_transform(df.A1.fillna("").str.replace('[\[\]\'\"]',""))
    print("Vectorizing abstracts")
    abstractVects = HashingVectorizer(n_features=2**15).fit_transform(df.N2.fillna("").str.replace('[\[\]\'\"]',""))
    # X = hstack((titleVects,keywordVects,authorVects,abstractVects))
    X = hstack((titleVects,authorVects,abstractVects))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X

def wsgi_application(env, response):
    print(env)

    if env.get("REQUEST_METHOD") == "POST" and env.get("PATH_INFO") == "/upload_train":
        print("uploading RIS file with labels")
        # print("{} bytes are uploaded".format(env.get("CONTENT_LENGTH")))
        inp = readInput(env)
        df = risFrom(inp)
        print(df.columns, df.shape)

        '''
        todo: add model training here
        '''
        X = risFeatures(df)
        print(X.shape)
        y = "todo: extract labels from ris"

        return respondWith('index.html', response)

    elif env.get("REQUEST_METHOD") == "POST" and env.get("PATH_INFO") == "/upload_pred":
        print("uploading RIS file without labels")
        inp = readInput(env)
        df = risFrom(inp)
        print(df.columns, df.shape)

        '''
        todo: add model inference here
        '''
        X = risFeatures(df)
        print(X.shape)

        return respondWith('show_preds.html', response)

    else:
        return respondWith('index.html', response)

host = "localhost"
port = 8080

bjoern.run(wsgi_application, host, port)
