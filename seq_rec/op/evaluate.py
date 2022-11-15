from typing import NamedTuple

from kfp.v2.dsl import (Input, Output, Dataset, Model, Metrics, component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def evaluate_op(
        model_dir: Input[Model],
        test_prep_ds_file: Input[Dataset],
        merchant_vocab_file: Input[Dataset],
        country_code: str,
        k: int,
        deployment_threshold: dict,
        # Comment the below metrics_output because there's a bug in Kubeflow Pipeline that prevents the condition start_training to work
        # metrics_output: Output[Metrics]
    ) -> NamedTuple("Outputs", [("dep_decision", str)]):
    """ Evaluate model performance and decide whether to deploy

    Args:
        model_dir (Input[Model])
        test_prep_ds_file (Input[Dataset])
        merchant_vocab_file (Input[Dataset]): used to calculate coverage
        country_code (str)
        k (int): number of predictions, usually needed to align with available
            serving signatures option from the saved model. For example, if
            there are only 2 signatures k=10 and k=100 then those are the only
            allowed values here
        deployment_threshold (dict): map of (metric_name, metric_value threshold).
            A deployment is allowed when the evaluation result
        metrics_output (Output[Metrics]): _description_

    Returns:
        dep_decision: deployment decision, "true" or "false"
    """
    import cloudpickle
    import tensorflow as tf

    from seq_rec.models.deeprec.ann_helper import ANNHelper
    from seq_rec.metrics import metrics

    BATCH_SIZE = 128

    test_prep_ds = tf.data.experimental.load(test_prep_ds_file.path)
    cached_ds = test_prep_ds.batch(BATCH_SIZE).cache()

    labels = list(test_prep_ds.map(lambda x: x['target_merchant_id']).as_numpy_iterator())
    test_prep_ds_input = ANNHelper.prepare_batch_input(test_prep_ds)
    with open(merchant_vocab_file.path, "rb") as f:
        merchant_vocab = cloudpickle.load(f)

    model = tf.saved_model.load(model_dir.path)
    serving_fn = model.signatures[f'k_{k}']
    recommendations = serving_fn(**test_prep_ds_input)
    rec_merchant_ids = recommendations['merchant_id'].numpy().tolist()
    rec_scores = recommendations['scores'].numpy().tolist()
    coverage = metrics.calc_coverage(rec_merchant_ids, catalog=merchant_vocab)
    accuracy_measures = metrics.calc_accuracy(rec_merchant_ids, labels)
    eval_result = {
        **accuracy_measures,
        'coverage': coverage
    }

    # Comment the below metrics_output logging part because there's a bug in Kubeflow Pipeline that prevents the condition start_training to work
    # for metric, value in eval_result.items():
    #     metrics_output.log_metric(metric, value)

    # metrics_output.metadata['country_code'] = country_code

    print(eval_result)

    deployment_checks = dict()
    for metric, threshold in deployment_threshold.items():
        metric_value = eval_result[metric]
        deployment_checks[metric] = metric_value >= threshold
    print("Deployment Checks:")
    print(deployment_checks)

    dep_decision = all(deployment_checks.values())
    if dep_decision:
        dep_decision = "true"
    else:
        dep_decision = "false"

    print(f"Deployment decision is {dep_decision}")

    return (dep_decision, )
