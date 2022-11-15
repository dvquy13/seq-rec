import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


def calc_accuracy(recommendations, labels, metrics: list = ['hit_rate']):
    """ Evaluate the recommendations against actual labels w.r.t. to a list of
        metrics

    Args:
        recommendations (2D array of M instances and k predictions)
        labels (2D array of M instances and undefined number of cols)
        metrics (List[str]): list of metrics to evaluate
    """
    output = dict()
    implemented_metrics = set(['hit_rate'])
    for metric in metrics:
        if metric == 'hit_rate':
            # TODO: Implement the calc_hit_rate
            _output = calc_hit_rate(recommendations, labels)
            output[metric] = _output
        else:
            logger.warning(f"No implementation for the metric {metric}")
            logger.warning(f"List of available metrics: {implemented_metrics}")

    return output


def calc_coverage(recommendations, catalog=None, labels=None):
    """ Percentage of items in the catalog that are recommended. Higher is better
        in the sense that more items are surfaced.

    Args:
        recommendations (2D array of M instances and k predictions)
        catalog (set): all the items in the catalog. If not provided then must input labels.
            If labels is also provided then ignore labels use catalog.
        labels (2D array of M instances and undefined number of cols). If not provided then must input catalog
    """
    logger.info("Calculating Coverage...")
    assert not(catalog is None and labels is None), "Need to provide either catalog or labels"
    recommended_items = set(np.array(recommendations).flatten())
    if catalog is None:
        catalog = np.array(labels).flatten()
    logger.info(f"# items in Catalog: {len(set(catalog))}")
    logger.info(f"# items in Recommended items: {len(recommended_items)}")
    logger.debug(f"Sample catalog: {list(catalog)[:10]}")
    logger.debug(f"Sample recommended_items: {list(recommended_items)[:10]}")
    intersected = set(catalog).intersection(recommended_items)

    return len(intersected) / len(catalog)


def calc_hit_rate(recommendations, labels):
    """ Hit rate is the how many times at least one of the label is in the recommendations.
        Assume the recommendations and labels fit into RAM.
        Mostly useful when there is only one positive in each label. If each label
        can have multiple positives then should use Precision At K instead.

    Args:
        recommendations (2D array of M instances and k predictions)
        labels (2D array of M instances and undefined number of cols)
    """
    logger.info("Calculating Hit Rate...")
    recommendations = np.array(recommendations)
    labels = np.array(labels)
    assert len(recommendations) == len(labels), \
        f"# of recommendations: {len(recommendations)} != # of labels {len(labels)}"
    recommendations_df = pd.Series(recommendations.tolist(), name='merchant_id').explode().reset_index()
    # Fix prediction score = 1 so that later can fill missing with 0
    recommendations_df['score'] = 1
    labels_df = pd.Series(labels.tolist(), name='merchant_id').explode().reset_index()
    hit_df = pd.merge(labels_df, recommendations_df, on=['index', 'merchant_id'], how='left').fillna(0)
    logger.debug(f"hit_df.head(): {hit_df.head()}")
    hit_cnt = hit_df['score'].sum()
    return hit_cnt / len(labels)
