# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

from ris import read_ris_lines

from ovr import labels_of, features_of, classify_cancer, clean_kws

import json


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


def model(request):
    """
    render model selection detail page.

    incomplete atm.

    :param request:
    :return:
    """
    metrics = json.load(open("multilabel_classification_metrics.json","rt"))
    print(metrics)
    def unzip_metric(metrics, key, t):
        precisions = filter(lambda p: p[1] >= t, metrics[key])
        pl = map(lambda p: p[0], precisions)
        pv = map(lambda p: p[1], precisions)
        return pl, pv

    # all precicions and recall
    pl, pv = unzip_metric(metrics, 'precision_score', 0.0)
    rl, rv = unzip_metric(metrics, 'recall_score', 0.0)

    context = {
        'precision_labels': pl,
        'precision_scores': pv,
        'recall_labels': rl,
        'recall_scores': rv
    }
    return render(request, 'cancer/model.html', context)


def ui_ris_upload_adapter(df, article_set):
    """
    processes the parsed dataframe, s.t. it becomes use-able by the json api.

    :param df: RIS parsed pandas.DataFrame
    :param article_set: either 'TRAIN' or 'INFERENCE'
    :return: {
        'body': 'json encoded data...'
    }
    """

    def concat_list_(values):
        if type(values) is list:
            return ','.join(values).decode('ascii', 'ignore')
        else:
            return ''

    import json

    articles = []

    # title
    df['T1'] = df.T1.map(concat_list_)
    # abstract
    df['N2'] = df.N2.map(concat_list_)

    for _, row in df.iterrows():
        entry = {
            'title': row.T1,
            'abstract': row.N2
        }

        if article_set == 'TRAIN':
            entry['keywords'] = row.KW

        articles.append(entry)

    # quick hack to emulate django request
    class AdapterRequest(object):
        def __init__(self, body):
            self.body = body

    serialized_body = json.dumps(
        {
            'articles': articles
        }
    )
    adapted_request = AdapterRequest(
        serialized_body
    )
    return adapted_request


def index(request):
    import cancer.json_api.endpoints

    results = []

    if 'train' in request.POST:
        print('train')
        df = df_from(request.FILES['file'].read())
        # train model, also dumps classifier and clear text label data to disk
        cancer.json_api.endpoints.train(
            ui_ris_upload_adapter(df, 'TRAIN')
        )
    elif 'test' in request.POST:
        df = df_from(request.FILES['file'].read())
        # predictions is an instance of JsonResponse django
        predictions = cancer.json_api.endpoints.inference(
            ui_ris_upload_adapter(df, 'INFERENCE')
        )
        predictions = json.loads(predictions.content)

        for article in predictions['article_predictions']:
            entry = {
                'index': article['article_id'],
                'title': article['title'],
                'labels': [
                    (kw['keyword'], '{0:.2f}'.format(100*float(kw['probability']))) for kw in article['keywords']
                ]
            }
            results.append(entry)
    else:
        print('GET')

    context = {
        'results': results,
    }

    return render(request, 'cancer/bs.html', context)
