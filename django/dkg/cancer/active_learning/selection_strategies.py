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
    def closest_to_decision_boundary(article_predictions, descending=False):
        """
        Return articles by average distances to decision boundaries.

        :param article_predictions: list of articles with keyword predictions.
        :param descending: sort article predictions by mean distance to hyperplane descending.
        :return: article_predictions sorted by mean distance to hyperplane.
        """
        for article in article_predictions:
            mean_distance = numpy.average(
                map(lambda entry: numpy.abs(entry['distance_to_hyperplane']), article['keywords'])
            )

            # if there are no keywords for this article, assign fallback value for sorting
            if numpy.isnan(mean_distance):
                article['mean_distance_to_hyperplane'] = 1e6
            else:
                article['mean_distance_to_hyperplane'] = mean_distance

        article_predictions.sort(
            key=lambda e: e['mean_distance_to_hyperplane'],
            reverse=descending
        )

        return article_predictions
