# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

ris_article_sources = (
    ('TRAIN', 'article is part of training set'),
    ('INFERENCE', 'article is part of inference set')
)


class RISArticle(models.Model):
    title = models.TextField()
    abstract = models.TextField()
    article_set = models.CharField(max_length=10, choices=ris_article_sources)


class RISArticleKeyword(models.Model):
    ris_article = models.ForeignKey(RISArticle, on_delete=models.CASCADE)
    keyword = models.TextField()
    annotator_name = models.TextField()
