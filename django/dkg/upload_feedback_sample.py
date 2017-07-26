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
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = {
        'article_id': 1,
        'keyword': '4-harn,pall-custom',
        'vote': 'OK', # or NOT OK to remove the label
        'annotator_name': 'annotator user name'
    }
    r = requests.post("http://---:---@localhost:8000/cancer/json/feedback", json=payload, headers=headers)
