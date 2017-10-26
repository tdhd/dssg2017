import django, scipy, json
import scipy as sp
from sklearn.model_selection import train_test_split
import cancer.persistence.models
from cancer.ris import read_ris_lines
from cancer.model_api.model import encode_features_of, encode_labels_of
from cancer.ovr import compute_scores, model_selection, compute_active_learning_curve

RIS_FILE = "/Users/felix/Code/Python/dssg2017/DSSG_StudyTagger_TestData/information-fuer-study-tagger.ris"

def concat_list_(values):
    """
    :param values: either list of string or np.nan
    :return: string repr of joined list of empty string
    """
    if type(values) is list:
        return ','.join(values).decode('ascii', 'ignore')
    else:
        return ''

def load_last_training_data(path=RIS_FILE):

    ris_df = read_ris_lines(open(path).readlines())

    ris_df['N2'] = ris_df.N2.map(concat_list_)
    ris_df['T1'] = ris_df.T1.map(concat_list_)
    ris_df['JA'] = ris_df.JA.map(concat_list_)
    ris_df['JF'] = ris_df.JF.map(concat_list_)
    ris_df['PB'] = ris_df.PB.map(concat_list_)
    ris_df['Y1'] = ris_df.Y1.map(concat_list_)

    # each keyword to three-tuple for compatability
    ris_df['KW'] = ris_df.KW.map(lambda kws: map(lambda kw: (kw, 1.0, 'test'), kws))

    X = encode_features_of(ris_df)
    y, label_names = encode_labels_of(ris_df)

    return X, y, label_names

def run_experiment(test_size=0.6, n_reps=5, percentage_samples=[1,2,5,10,15,30,50,100]):
    '''
    Runs a multilabel classification experiment
    '''
    print("Loading data")
    X, y, label_names = load_last_training_data()

    X_train, X_tolabel, y_train, y_tolabel = train_test_split(X, y, test_size=test_size)

    X_test, X_validation, y_test, y_validation = train_test_split(X_tolabel, y_tolabel, test_size=(1-test_size))
    print("Model Selection")
    # do model selection on training data
    clf = model_selection(X_train, y_train)

    # compute active learning curves
    active_learning_curves, random_learning_curves, baseline_lows, baseline_highs = [],[],[],[]
    for irep in range(n_reps):
        X_train, X_tolabel, y_train, y_tolabel = train_test_split(X, y, test_size=test_size)
        X_test, X_validation, y_test, y_validation = train_test_split(X_tolabel, y_tolabel, test_size=(1-test_size))
        active_learning_curve, random_learning_curve, baseline_low, baseline_high = compute_active_learning_curve(X_train, y_train, X_test, y_test, X_validation, y_validation, clf,percentage_samples=percentage_samples)
        active_learning_curves.append(active_learning_curve)
        random_learning_curves.append(random_learning_curve)
        baseline_lows.append(baseline_low)
        baseline_highs.append(baseline_high)
    results = {
        'active_learning_curves':active_learning_curves,
        'random_learning_curves':random_learning_curves,
        'baseline_lows':baseline_lows,
        'baseline_highs':baseline_highs,
        'percentage_samples':percentage_samples
        }
    json.dump(results,open("active_learning_curves.json","wt"))
    return active_learning_curves, random_learning_curves, baseline_lows, baseline_highs

def compute_active_learning_curve(X_train,y_train,X_test,y_test,X_validation, y_validation, clf,percentage_samples=[1,2,5,10,15,30,50,100]):
    '''
    Emulate active learning with annotators:
    for a given training, test and validation set, get the validation error by
    training on training data only, then the score when trained on training and
    test data and then the increasing validation score when adding more labelled
    data, either with random selection or with active learning. The results are
    the increase in scores with the respective sampling policy
    '''
    print('Computing active learning curve:')
    clf = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'), n_jobs=-1).fit(X_train, y_train)
    baseline_low = label_ranking_average_precision_score(y_validation.toarray(), clf.predict_proba(X_validation))
    clf_trained = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'), n_jobs=-1).fit(vstack([X_train, X_test]), vstack([y_train, y_test]))
    baseline_high = label_ranking_average_precision_score(y_validation.toarray(), clf_trained.predict_proba(X_validation))
    print('\tBaseline on test: {}, baseline score on train and test {}'.format(baseline_low, baseline_high))

    # score test data for active learning sorting
    label_probas = clf.predict_proba(X_test)

    # run a random sampling procedure for training with increasing amounts of labels
    random_priorities = np.random.permutation(label_probas.shape[0])

    random_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[random_priorities[:n_samples],:]
        y_labelled = y_test[random_priorities[:n_samples],:]
        clf_current = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'), n_jobs=-1).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
        current_score = label_ranking_average_precision_score(y_validation.toarray(), clf_current.predict_proba(X_validation))
        print('\t(RANDOM) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        random_learning_curve.append(current_score)

    # mean distance to hyperplane
    dists = abs(logit(label_probas)).mean(axis=1)
    # run active learning procedure for training with increasing amounts of labels
    priorities = dists.argsort()

    active_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[priorities[:n_samples],:]
        y_labelled = y_test[priorities[:n_samples],:]
        clf_current = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'),n_jobs=-1).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
        current_score = label_ranking_average_precision_score(y_validation.toarray(), clf_current.predict_proba(X_validation))
        print('\t(ACTIVE LEARNING) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        active_learning_curve.append(current_score)

    return active_learning_curve, random_learning_curve, baseline_low, baseline_high

def plot_results(fn):
    import pylab
    results = json.load(open(fn))
    ac = sp.vstack(results['active_learning_curves'])
    rc = sp.vstack(results['random_learning_curves'])
    pylab.figure(figsize=(10,10))
    pylab.hold('all')
    pylab.plot([0.5,.8],[0.5,.8],'k-')
    for i in range(len(results['percentage_samples'])):
        pylab.plot(ac[:,i],rc[:,i],'o')
    pylab.xlim([.64,.7])
    pylab.ylim([.64,.7])
    pylab.legend([0]+results['percentage_samples'])
    pylab.xlabel("Active Learning")
    pylab.ylabel("Random")
    pylab.title("Classifier score as function of n_samples")
    pylab.savefig('active_learning_curves.pdf')
