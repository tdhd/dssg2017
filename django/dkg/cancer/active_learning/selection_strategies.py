import numpy


class SelectionStrategies(object):

    @staticmethod
    def default(article_predictions):
        return SelectionStrategies.closest_to_decision_boundary(article_predictions)

    @staticmethod
    def random(article_predictions):
        """
        Return random ordering of article predictions.

        :param article_predictions: list of articles with keyword predictions.
        :return: randomly shuffled article_predictions.
        """
        numpy.random.shuffle(article_predictions)
        return article_predictions

    @staticmethod
    def closest_to_decision_boundary(article_predictions, ascending=True):
        """
        Return articles by average distances to decision boundaries.

        :param article_predictions: list of articles with keyword predictions.
        :param ascending: sort article predictions by mean distance to hyperplane ascending.
        :return: article_predictions sorted by mean distance to hyperplane.
        """
        for article in article_predictions:
            mean_distance = numpy.average(
                map(lambda entry: numpy.abs(entry['distance_to_hyperplane']), article['keywords'])
            )
            article['mean_distance_to_hyperplane'] = mean_distance

        article_predictions.sort(
            key = lambda e: e['mean_distance_to_hyperplane'],
            reverse=ascending
        )
        return article_predictions
