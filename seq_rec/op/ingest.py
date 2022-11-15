from typing import NamedTuple

from kfp.v2.dsl import (Dataset, Output, component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def detect_resume_training_op(
        resume_training: bool,
        checkpoint_bucket: str,
        last_checkpoint_date_blob_name: str
) -> NamedTuple("Outputs", [("resume_training", bool), ("last_checkpoint_date", str), ("start_training", str)]):
    """ Decide whether to resume training on top of previous runs

    Args:
        resume_training (bool): Pipeline input param
        checkpoint_bucket (str): Name of the GCS bucket that host the checkpoint directory
        last_checkpoint_date_blob_name (str): The path to the last checkpoint date file in the bucket
    """
    from google.cloud import storage
    from datetime import datetime

    print(f"last_checkpoint_date_blob_name: {last_checkpoint_date_blob_name}")

    if not resume_training:
        return (False, "", "true")

    gcs_client = storage.Client()

    bucket = gcs_client.get_bucket(checkpoint_bucket)
    blob = bucket.blob(last_checkpoint_date_blob_name)

    if blob.exists():
        blob = bucket.get_blob(last_checkpoint_date_blob_name)

        with blob.open(mode='r') as f:
            last_checkpoint_date_str = f.read()
            if last_checkpoint_date_str:
                print(f"last_checkpoint_date_str: {last_checkpoint_date_str}")
                job_current_date = datetime.now().strftime("%Y-%m-%d")
                if job_current_date == last_checkpoint_date_str:
                    return (True, last_checkpoint_date_str, "false")
                return (True, last_checkpoint_date_str, "true")

    return (False, "", "true")


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def tf_download_bq_table_op(
        raw_train_ds_file: Output[Dataset],
        raw_test_ds_file: Output[Dataset],
        raw_full_ds_file: Output[Dataset],
        resume_training: bool = True,
        last_checkpoint_date: str = "",
        country_code: str = 'SG'
    ):
    """ Using Tensorflow API to download BigQuery tables

    Args:
        raw_train_ds_file (Output[Dataset])
        raw_test_ds_file (Output[Dataset])
        raw_full_ds_file (Output[Dataset])
        resume_training (bool, optional): whether to load only the new data from previous train. Defaults to True.
        country_code (str)
    """
    import tensorflow as tf

    import seq_rec.utils as utils
    import seq_rec.io as io

    overrides=[
        f"env.io.training.applied_country_code=['{country_code}']"
    ]
    # Uncomment for testing cause this is less data
    # overrides.append("env.io.training.bigquery.train_table_id=fct_seq_rec_build_output_training_pad_7d")

    if resume_training:
        _overrides = [
            "env.io.training.bigquery.train_table_id=fct_seq_rec_build_output_training_pad_checkpoint_train",
            "env.io.training.bigquery.test_table_id=fct_seq_rec_build_output_training_pad_checkpoint_test",
            "env.io.training.bigquery.full_table_id=fct_seq_rec_build_output_training_pad",
        ]
        print(f"Resume Training is enabled. Extending config with: {_overrides}")
        overrides.extend(_overrides)

    cfg = utils.load_cfg(overrides=overrides)

    train_ds = io.load_training_data(cfg.env.io.training, part='train', download=True, last_checkpoint_date=last_checkpoint_date)
    test_ds = io.load_training_data(cfg.env.io.training, part='test')
    full_ds = io.load_training_data(cfg.env.io.training, part='full')

    tf.data.experimental.save(train_ds, raw_train_ds_file.path)
    tf.data.experimental.save(test_ds, raw_test_ds_file.path)
    tf.data.experimental.save(full_ds, raw_full_ds_file.path)
