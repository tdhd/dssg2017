import django
from sklearn.model_selection import train_test_split
import cancer.persistence.models
from cancer.ris import read_ris_lines
from cancer.model_api.model import encode_features_of, encode_labels_of
from cancer.ovr import compute_scores, model_selection, compute_active_learning_curve
import scipy

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

def run_experiment(test_size=0.6, n_reps=5):
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
        active_learning_curve, random_learning_curve, baseline_low, baseline_high = compute_active_learning_curve(X_train, y_train, X_test, y_test, X_validation, y_validation, clf)
        active_learning_curves.append(active_learning_curve)
        random_learning_curves.append(random_learning_curve)
        baseline_lows.append(baseline_low)
        baseline_highs.append(baseline_high)
    return active_learning_curves, random_learning_curves, baseline_lows, baseline_highs
