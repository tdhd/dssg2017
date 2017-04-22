# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render

from ris import read_ris_lines

def KW_stats_from(df):
    import pandas as pd
    import numpy as np
    df['split_KW'] = df.KW.map(lambda r: r.split(','))
    import itertools
    kw = pd.DataFrame({'kw': list(itertools.chain(*df['split_KW']))})
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
    latest_question_list = [1, 2, 3]#Question.objects.order_by('-pub_date')[:5]
    context = {'latest_question_list': latest_question_list}
    return render(request, 'cancer/index.html', context)

def upload_train(request):
    df = df_from(request.FILES['file'].read())

    KW_stats_from(df)

    p_ausschluss = 100.0*df.KW.str.contains('Ausschluss').sum()/df.shape[0]
    p_basis = 100.0*df.KW.str.contains('basis').sum()/df.shape[0]
    print(p_basis)
    return HttpResponse('uploaded file with {} articles, {}% with Ausschluss'.format(df.shape[0], p_ausschluss))

def upload_pred(request):
    df = df_from(request.FILES['file'].read())
    context = {} #{'latest_question_list': [1, 2, 3]}
    return render(request, 'cancer/preds.html', context)
