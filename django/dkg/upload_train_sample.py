from cancer.ris import read_ris_lines
import requests


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

    ris_df = ris_df.head()

    ris_df['joined_titles'] = ris_df.T1.map(concat_list_)
    ris_df['joined_abstracts'] = ris_df.N2.map(concat_list_)

    parsed_articles = [{'title': row.joined_titles, 'abstract': row.joined_abstracts, 'keywords': row.KW} for _, row in ris_df.iterrows()]

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = {
        'articles': parsed_articles
    }
    r = requests.post("http://---:---@localhost:8000/cancer/json/train", json=payload, headers=headers)
