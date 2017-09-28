import numpy


class SelectionStrategies(object):

    @staticmethod
    def default(article_predictions):
        return SelectionStrategies.uncertainty_sampling(article_predictions)

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
    def uncertainty_sampling(article_predictions, descending=False):
        """
        Return articles by average distances to decision boundaries.

        :param article_predictions: list of articles with keyword predictions.
        :param descending: sort article predictions by mean distance to hyperplane descending.
        :return: article_predictions sorted by mean distance to hyperplane.
        """
        for article in article_predictions:
            min_distance = 1e6
            try:
                min_distance = min(
                    map(lambda entry: numpy.abs(entry['distance_to_hyperplane']), article['keywords'])
                )
            except:
                pass

            article['min_distance_to_hyperplane'] = min_distance

        article_predictions.sort(
            key=lambda e: e['min_distance_to_hyperplane'],
            reverse=descending
        )

        return article_predictions
