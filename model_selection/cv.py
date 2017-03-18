from sklearn.model_selection import cross_val_score, GridSearchCV
#from sklearn import svm
#from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np

def cv_debug(clf, X, y, grid, k, scorer):
    '''
    prints the model selection results with out of fold estimates

    example parameters:
    vec = TfidfVectorizer(max_features=32)
    logreg = LogisticRegression()
    clf = Pipeline([('tfidf', vec), ('logreg', logreg)])
    y = features.useful.apply(lambda r: bool(r))
    X = features.abstract.fillna("")

    def roc_auc_scorer(y_true, y_pred):
        return roc_auc_score(y_true, y_pred[:, 1])
    scorer = make_scorer(roc_auc_scorer, needs_proba=True)

    grid = {
        'tfidf__max_features': [512, 1024, 2048, 4096],
        'logreg__C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    }
    
    cv_debug(clf, X, y, grid, k=3,scorer=scorer)
    '''

    #scorer = make_scorer(roc_auc_scorer, needs_proba=True)

    cv = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=k, scoring=scorer, verbose=1)
    cv.fit(X, y)

    means = cv.cv_results_['mean_test_score']
    stds = cv.cv_results_['std_test_score']
    params = np.array(cv.cv_results_['params'])

    idcs = means.argsort()
    means = means[idcs]
    stds = stds[idcs]
    paramss = params[idcs]

    for mean, std, params in zip(means, stds, paramss):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
