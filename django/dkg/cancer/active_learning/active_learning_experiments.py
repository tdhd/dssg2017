<<<<<<< HEAD
# import django
import json
import scipy as sp
import pandas as pd
import itertools
=======
import django, scipy, json
import scipy as sp
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
from sklearn.model_selection import train_test_split
import cancer.persistence.models
from cancer.ris import read_ris_lines
from cancer.model_api.model import encode_features_of, encode_labels_of
from cancer.ovr import compute_scores, model_selection, compute_active_learning_curve

<<<<<<< HEAD
RIS_FILE = "/Users/felix/Code/Python/dssg2017/DSSG_StudyTagger_TestData_20170731/train_Data_wholeDB_20170731.ris"
IGNORE_KEYWORDS = "/Users/felix/Code/Python/dssg2017/DSSG_StudyTagger_TestData_20170731/zu_ignorierende_keywords.txt"
=======
RIS_FILE = "/Users/felix/Code/Python/dssg2017/DSSG_StudyTagger_TestData/information-fuer-study-tagger.ris"
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2

def concat_list_(values):
    """
    :param values: either list of string or np.nan
    :return: string repr of joined list of empty string
    """
    if type(values) is list:
        return ','.join(values).decode('ascii', 'ignore')
    else:
        return ''

<<<<<<< HEAD
def load_last_training_data(path=RIS_FILE, subsample=0.2):

    ris_df = read_ris_lines(open(path).readlines()).sample(frac=subsample)
=======
def load_last_training_data(path=RIS_FILE):

    ris_df = read_ris_lines(open(path).readlines())
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2

    ris_df['N2'] = ris_df.N2.map(concat_list_)
    ris_df['T1'] = ris_df.T1.map(concat_list_)
    ris_df['JA'] = ris_df.JA.map(concat_list_)
    ris_df['JF'] = ris_df.JF.map(concat_list_)
    ris_df['PB'] = ris_df.PB.map(concat_list_)
    ris_df['Y1'] = ris_df.Y1.map(concat_list_)
<<<<<<< HEAD
    # kws = pd.Series(",".join(itertools.chain(*ris_df.KW)).split(",")).str.strip().value_counts()
    # rare = kws[kws < 100].index.tolist()
    rare = []
    # remove to-be-ignored-keywords

    ignore = [x.strip() for x in open(IGNORE_KEYWORDS).readlines()] + rare
    print("Removing {} keywords".format(len(ignore)))

    def ignore_words(words, ignorewords):
        return [w.strip() for w in words if not any([iw in w for iw in ignorewords])]

    ris_df['KW'] = ris_df['KW'].apply(lambda x: ignore_words(x, ignore))
    ris_df = ris_df[ris_df.KW.apply(lambda x: len(x)>0)]

    print("Extracted {} samples".format(len(ris_df)))
=======
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2

    # each keyword to three-tuple for compatability
    ris_df['KW'] = ris_df.KW.map(lambda kws: map(lambda kw: (kw, 1.0, 'test'), kws))

    X = encode_features_of(ris_df)
    y, label_names = encode_labels_of(ris_df)

    return X, y, label_names

<<<<<<< HEAD
def run_experiment(test_size=0.8, n_reps=5, percentage_samples=[1,2,5,10,15,30,50,100],subsample=.1):
=======
def run_experiment(test_size=0.6, n_reps=5, percentage_samples=[1,2,5,10,15,30,50,100]):
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
    '''
    Runs a multilabel classification experiment
    '''
    print("Loading data")
<<<<<<< HEAD
    X, y, label_names = load_last_training_data(subsample=subsample)
=======
    X, y, label_names = load_last_training_data()
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2

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

def plot_results(fn):
    import pylab
    results = json.load(open(fn))
    ac = sp.vstack(results['active_learning_curves'])
    rc = sp.vstack(results['random_learning_curves'])
    pylab.figure(figsize=(10,10))
    pylab.hold('all')
<<<<<<< HEAD
    # pylab.hold(True)
    pylab.plot([0.5,.8],[0.5,.8],'k-')
    for i in range(len(results['percentage_samples'])):
        pylab.plot(ac[:,i],rc[:,i],'o')
    pylab.xlim([rc.min()-.05,ac.max()+.05])
    pylab.ylim([rc.min()-.05,ac.max()+.05])
=======
    pylab.plot([0.5,.8],[0.5,.8],'k-')
    for i in range(len(results['percentage_samples'])):
        pylab.plot(ac[:,i],rc[:,i],'o')
    pylab.xlim([.64,.7])
    pylab.ylim([.64,.7])
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
    pylab.legend([0]+results['percentage_samples'])
    pylab.xlabel("Active Learning")
    pylab.ylabel("Random")
    pylab.title("Classifier score as function of n_samples")
    pylab.savefig('active_learning_curves.pdf')
