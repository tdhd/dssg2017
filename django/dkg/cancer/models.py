# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models


class RISArticle(models.Model):
    title = models.TextField()
    abstract = models.TextField()
    is_training_set = models.BooleanField()


class RISArticleKeyword(models.Model):
    ris_article = models.ForeignKey(RISArticle, on_delete=models.CASCADE)
    keyword = models.TextField()
    annotater_name = models.TextField()
