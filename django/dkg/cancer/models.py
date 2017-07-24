# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

ris_article_sources = (
    ('TRAIN', 'article is part of training set'),
    ('INFERENCE', 'article is part of inference set')
)


def all_articles_by(article_set):
    """
    Returns all joined articles with tab-separated keywords in one column.
    One article per row.
    """
    return RISArticle.objects.raw(
        """
        SELECT
            a.id,
            MAX(a.title) title,
            MAX(a.abstract) abstract,
            MAX(a.article_set) article_set,
            GROUP_CONCAT(akw.keyword, '\t') ts_keywords,
            GROUP_CONCAT(akw.keyword_probability, '\t') ts_keyword_probabilities
        FROM cancer_risarticle a
        INNER JOIN cancer_risarticlekeyword akw
            ON akw.ris_article_id == a.id
        WHERE
            a.article_set = '{}'
        GROUP BY
            a.id
        """.format(article_set)
    )


def insert_with_keywords(articles_with_keywords, article_set):
    """
    inserts articles with their respective keywords.
    :param articles_with_keywords: n-element list of dicts for articles with keywords.
    :param article_set: either 'TRAIN' or 'INFERENCE',
    :return:
    """
    from django.db import connection

    # FIXME: need proper escaping
    select_article_statements = [
        "SELECT {}, '{}', '{}', '{}'".format(
            id,
            article['title'].replace("'", " "),
            article['abstract'].replace("'", " "),
            article_set
        )
        for id, article in enumerate(articles_with_keywords)
    ]

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    insert_batch_size = 100
    with connection.cursor() as cursor:
        for chunked_selects in chunks(select_article_statements, insert_batch_size):
            chunked_selects = ' UNION ALL '.join(chunked_selects)
            insert_statement = """INSERT INTO 'cancer_risarticle' ('id', 'title', 'abstract', 'article_set') {};""".format(chunked_selects)
            cursor.execute(insert_statement)

        # generate n-element insert for n batches articles
        for chunked_articles_with_keywords in chunks(list(enumerate(articles_with_keywords)), 35):
            selects = []
            for article_id, article in chunked_articles_with_keywords:
                # FIXME: need proper escaping
                article_keyword_selects = [
                    "SELECT {}, '{}', {}, '{}'".format(
                        article_id,
                        keyword.replace("'", " "),
                        proba,
                        article_set
                    ) for keyword, proba in zip(article['keywords'], article['keyword_probabilities'])
                ]
                selects += article_keyword_selects
            print len(selects)
            selects = ' UNION ALL '.join(selects)
            insert_statement = """
                INSERT INTO 'cancer_risarticlekeyword'
                    ('ris_article_id', 'keyword', 'keyword_probability', 'annotator_name')
                {};
            """.format(selects)
            cursor.execute(insert_statement)
            print 'executed'


class RISArticle(models.Model):
    title = models.TextField()
    abstract = models.TextField()
    article_set = models.CharField(max_length=10, choices=ris_article_sources)


class RISArticleKeyword(models.Model):
    ris_article = models.ForeignKey(RISArticle, on_delete=models.CASCADE)
    keyword = models.TextField()
    keyword_probability = models.FloatField() # in [0, 1]
    annotator_name = models.TextField()
