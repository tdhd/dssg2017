import cancer.model_api.model
import cancer.ovr
from cancer.ris import read_ris_lines


def concat_list_(values):
    """
    :param values: either list of string or np.nan
    :return: string repr of joined list of empty string
    """
    if type(values) is list:
        return ','.join(values).decode('ascii', 'ignore')
    else:
        return ''


if __name__ == "__main__":

    ris_df = read_ris_lines(
        open('../../data/information-fuer-study-tagger.ris').readlines()
    )

    ris_df['N2'] = ris_df.N2.map(concat_list_)
    ris_df['T1'] = ris_df.T1.map(concat_list_)
    ris_df['JA'] = ris_df.JA.map(concat_list_)
    ris_df['JF'] = ris_df.JF.map(concat_list_)
    ris_df['PB'] = ris_df.PB.map(concat_list_)
    ris_df['Y1'] = ris_df.Y1.map(concat_list_)

    # each keyword to three-tuple for compatability
    ris_df['KW'] = ris_df.KW.map(lambda kws: map(lambda kw: (kw, 1.0, 'test'), kws))

    X = cancer.model_api.model.encode_features_of(ris_df)
    Y, label_names = cancer.model_api.model.encode_labels_of(ris_df)

    print X.shape, Y.shape

    clf = cancer.ovr.classify_cancer(X, Y, label_names)
