from typing import NamedTuple

from kfp.v2.dsl import (Input, Output, HTML, Artifact, Dataset, component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.tfdv_image_uri)
def generate_tfdv_schema_op(
        X_train_file: Input[Dataset],
        X_train_schema_file: Output[Artifact]
    ):
    import tensorflow_data_validation as tfdv

    train_stats = tfdv.generate_statistics_from_tfrecord(data_location=X_train_file.path)
    schema = tfdv.infer_schema(statistics=train_stats)
    tfdv.write_schema_text(schema, X_train_schema_file.path)


@component(base_image=cfg.env.pipeline.kubeflow.tfdv_image_uri)
def validate_tfdv_schema_op(
        X_train_schema_base_file: Input[Artifact],
        X_train_file: Input[Dataset],
        validation_file: Output[HTML]
    ) -> NamedTuple("Outputs", [("is_abnormal", str)]):
    import tensorflow_data_validation as tfdv
    import tensorflow as tf

    X_train_schema_base = tfdv.load_schema_text(X_train_schema_base_file.path)

    train_stats = tfdv.generate_statistics_from_tfrecord(data_location=X_train_file.path)

    anomalies = tfdv.validate_statistics(statistics=train_stats, schema=X_train_schema_base)
    # Could not get the HTML output of tfdv.display_anomalies so manually copy from source and modify
    # Ref: https://github.com/tensorflow/data-validation/blob/5cea7668feb75538f44122a857dc799592bc5d09/tensorflow_data_validation/utils/display_util.py#L230

    from tensorflow_data_validation.utils.display_util import get_anomalies_dataframe
    anomalies_df = get_anomalies_dataframe(anomalies)

    if anomalies_df.empty:
        anomalies_html = '<h4 style="color:green;">No anomalies found.</h4>'
        is_abnormal = "false"
    else:
        anomalies_html = anomalies_df.to_html()
        is_abnormal = "true"

    with open(validation_file.path, "w") as f:
        f.write(anomalies_html)

    return (is_abnormal, )
