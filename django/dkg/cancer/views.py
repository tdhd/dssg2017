# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json

import cancer.persistence.models
import django.conf
from django.http import HttpResponse
from django.shortcuts import render

from ris import read_ris_lines, write_ris_lines


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

    def fillna_list(values):
        if type(values) is list:
            return values
        else:
            return []

    import json

    articles = []

    # how to find columns that have more than either 0 or 1 value in them:
    # In[14]: def lenlist(values):
    #     ...:     if type(values) == list:
    #     ...:         return len(values)
    #     ...:     else:
    #     ...:         return 0
    # lens = {col: df[col].map(lenlist).unique() for col in df.columns}

    one_to_n_columns = [
        'A1',  # primary author
        'A2',  # secondary author
        'KW'   # keywords
    ]

    # list -> string for all columns except for primary/secondary authors and keywords
    for col in df.columns:
        if col not in one_to_n_columns:
            df[col] = df[col].map(concat_list_)
        else:
            df[col] = df[col].map(fillna_list)

    for _, row in df.iterrows():
        entry = row.to_dict()
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
                'abstract': article['abstract'],
                'labels': [
                    (kw['keyword'], '{0:.2f}'.format(100*float(kw['probability']))) for kw in article['keywords']
                ]
            }
            results.append(entry)
    else:
        print('GET')

    context = {
        'results': results,
        'json_api': {
            'username': django.conf.settings.USER_NAME,
            'password': django.conf.settings.PASSWORD,
            'port': django.conf.settings.PORT
        }
    }

    return render(request, 'cancer/bs.html', context)


def download_test_ris(request):
    """
    loads inference pandas dataframe pickle, converts to RIS and sends RIS file contents to client.
    """
    current_inference_filename = cancer.persistence.models.latest_persistence_filename(django.conf.settings.INFERENCE_ARTICLES_PATTERN)
    persistence = cancer.persistence.models.PandasPersistence(
        current_inference_filename
    )
    articles = persistence.load_data()

    write_ris_lines(
        django.conf.settings.INFERENCE_ARTICLES_RIS_PATH,
        articles
    )

    with open(django.conf.settings.INFERENCE_ARTICLES_RIS_PATH, 'r') as f:
        contents = f.read()
        response = HttpResponse(
            contents, content_type='application/force-download'
        )
        # response['X-Sendfile'] = smart_str(django.conf.settings.INFERENCE_ARTICLES_RIS_PATH)
        response['Content-Disposition'] = 'attachment; filename=inference.ris'
        # It's usually a good idea to set the 'Content-Length' header too.
        # You can also set any other required headers: Cache-Control, etc.
        return response
