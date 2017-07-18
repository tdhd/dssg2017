import numpy


class SelectionStrategies(object):

    @staticmethod
    def random(classifier, X):
        """
        Compute random ordering of X indices.

        :param classifier: scikit estimator.
        :param X: numpy array of shape (n, m).
        :return: n-element list of indices.
        """
        idcs = list(range(X.shape[0]))
        numpy.random.shuffle(idcs)
        return idcs

    @staticmethod
    def closest_to_decision_boundary(classifier, X):
        """
        Compute ordering of X indices by distance to hyperplanes.

        :param classifier: scikit estimator.
        :param X: numpy array of shape (n, m).
        :return: n-element list of indices.
        """
        pass
