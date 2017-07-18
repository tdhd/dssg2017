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


class RISArticle(models.Model):
    title = models.TextField()
    abstract = models.TextField()
    article_set = models.CharField(max_length=10, choices=ris_article_sources)


class RISArticleKeyword(models.Model):
    ris_article = models.ForeignKey(RISArticle, on_delete=models.CASCADE)
    keyword = models.TextField()
    keyword_probability = models.FloatField() # in [0, 1]
    annotator_name = models.TextField()
