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
        'article_id': 469,
        'keywords': ['4-harn,pall-custom', '1-med'],
        'annotator_name': 'annotator user name'
    }
    r = requests.post("http://---:---@localhost:8000/cancer/json/feedback_batch", json=payload, headers=headers)
