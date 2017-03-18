import pandas as pd
import numpy as np
#%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion

df = pd.read_csv('../../datadiveberlin/features/features.csv')

from sklearn.base import BaseEstimator, TransformerMixin
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.
#http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class Pandify(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return pd.DataFrame(data)
 


df['abstract'] = df.abstract.fillna(" ")
X = df
y = df.useful

ctv = CountVectorizer()
lr = LogisticRegression()

pipeline = Pipeline([
    
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the abstract
            ('abstract', Pipeline([
                ('selector', ItemSelector(key='abstract')),
                ('ctv', ctv),
            ])),

            # Pipeline for pulling ad hoc features from post's body
            ('other', Pipeline([
                ('selector', ItemSelector(key='review_article')),
                ('pdd', Pandify())
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'abstract': 0.8,
            'other': 0.5,
        },
    )),

    # Use a SVC classifier on the combined features
    ('lr', lr),
])

pipeline.set_params(union__abstract__ctv__analyzer='word')
scores = cross_val_score( pipeline, X, y, cv=5, scoring='accuracy')
print(scores)

