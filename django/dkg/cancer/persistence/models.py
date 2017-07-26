import pandas as pd


class Persistence(object):
    def __init__(self, save_path):
        self.save_path = save_path

    def save_batch(self, elements):
        """
        :param elements: n-element list of dicts to be saved.
        :return:
        """
        raise NotImplementedError()

    def load_data(self):
        """
        loads data, articles with keywords.
        :return: pandas.DataFrame with columns: WRITEME
        """
        raise NotImplementedError()

    def update(self, data):
        """
        overwrite self.save_path with current data.
        :param data: pandas.DataFrame with columns: WRITEME
        :return:
        """
        raise NotImplementedError()


class PandasPersistence(Persistence):
    def __init__(self, save_path):
        super(PandasPersistence, self).__init__(save_path)

    def save_batch(self, elements):
        import pandas as pd
        pd.DataFrame(elements).to_pickle(self.save_path)

    def load_data(self):
        import pandas as pd
        return pd.read_pickle(self.save_path)

    def update(self, data):
        data.to_pickle(self.save_path)
