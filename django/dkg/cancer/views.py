# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render

from ris import read_ris_lines

from ovr import labels_of, features_of, classify_cancer, clean_kws

from sklearn.externals import joblib
import json
import pickle
import numpy as np

def KW_stats_from(df):
    import pandas as pd
    import numpy as np
    df['split_KW'] = df.KW.map(clean_kws)
    import itertools
    kw = pd.DataFrame({'kw': list(itertools.chain(*df['split_KW']))})
    kw['kw'] = kw['kw'].map(lambda k: k.lower())
    print(kw)
    agg = kw.groupby('kw').size().reset_index().sort_values(0, ascending=False)
    agg['n'] = agg[0]
    del agg[0]
    agg['np'] = agg['n']/agg['n'].sum()
    agg['csum'] = np.cumsum(agg['np'])
    print(agg.head(150))
    print('{} distinct labels'.format(agg.shape[0]))

def df_from(ris_contents):
    ris_lines = ris_contents.decode('ascii', 'ignore').split('\n')
    df = read_ris_lines(ris_lines)
    # df.to_csv('uploaded.csv', index=False)
    # print(df.head())
    return df

def index(request):
    # print(request.body)
    clf_filename = 'clf.pkl'
    labels_filename = 'labels.pkl'
    results = []
    if 'train' in request.POST:
        print('train')
        df = df_from(request.FILES['file'].read())
        KW_stats_from(df)
        # df = df[0:15000]
        X = features_of(df)
        y, label_names = labels_of(df, 'KW')
        clf = classify_cancer(X, y, label_names)
        joblib.dump(clf, clf_filename)
        with open(labels_filename, 'w') as f:
            # f.write(json.dumps(label_names))
            pickle.dump(label_names, f)
        print(clf)
    elif 'test' in request.POST:

        clf = joblib.load(clf_filename)
        print(clf)

        # TODO: use feature pipeline and do not encode independently

        df = df_from(request.FILES['file'].read())
        X = features_of(df)

        with open(labels_filename, 'r') as f:
            label_names = pickle.load(f)

        y = clf.predict_proba(X)[0:100]

        '''
        TODO:
            - also return precisions for all labels
        '''

        results = []
        for idx in range(y.shape[0]):
            row = y[idx,:]
            title = df.loc[idx,'T1']
            labels_with_probas = [(label_names[l], np.round(row[l], 2)) for l in row.argsort()[::-1]]
            labels_with_probas = filter(lambda lp: lp[1] > 0.01, labels_with_probas)
            result = {
                'index': idx,
                'title': title,
                'labels': labels_with_probas
            }
            results.append(result)
    else:
        print('GET')

    context = {'results': results}
    return render(request, 'cancer/bs.html', context)

# def upload_train(request):
#     df = df_from(request.FILES['file'].read())
#
#     KW_stats_from(df)
#
#     p_ausschluss = 100.0*df.KW.str.contains('Ausschluss').sum()/df.shape[0]
#     p_basis = 100.0*df.KW.str.contains('basis').sum()/df.shape[0]
#     print(p_basis)
#
#     print(df['N2'].head())
#     df = df[0:5000]
#
#     X = features_of(df)
#     y, label_names = labels_of(df, 'KW')
#     clf = classify_cancer(X, y, label_names)
#     print(clf)
#     return HttpResponse('uploaded file with {} articles, {}% with Ausschluss'.format(df.shape[0], p_ausschluss))
#
# def upload_pred(request):
#     df = df_from(request.FILES['file'].read())
#     context = {} #{'latest_question_list': [1, 2, 3]}
#     return render(request, 'cancer/preds.html', context)
