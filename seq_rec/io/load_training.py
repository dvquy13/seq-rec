import os
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

from google.cloud import bigquery
import tensorflow as tf
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession
from tensorflow.python.framework import dtypes

import numpy as np

from .trigger_build import update_materialized_table


def download_bigquery_table(
        project_id: str,
        dataset_id: str,
        table_id: str,
        shuffle_buffer_size: int = 1_000_000
    ):
    # Ref: https://www.tensorflow.org/io/api_docs/python/tfio/bigquery/BigQueryClient
    SELECTED_FIELDS_SCHEMA = {
        "user_id": {
            "output_type": dtypes.string
        },
        "event_name": {
            "output_type": dtypes.string
        },
        "merchant_id": {
            "output_type": dtypes.string
        },
        "prev_search_term_list": {
            "output_type": dtypes.string
        },
        "prev_event_ruid_list": {
            "output_type": dtypes.string
        },
        "recent_txn_ruid_list": {
            "output_type": dtypes.string
        },
        "prev_search_term_time_diff_seconds_list": {
            "output_type": dtypes.string
        },
        "prev_event_time_diff_seconds_list": {
            "output_type": dtypes.string
        },
        "recent_txn_time_diff_days_list": {
            "output_type": dtypes.string
        },
        "cnt_not_null_context_search_term": {
            "output_type": dtypes.int64  # use dtypes.int32 will lead to kerner shut down...
        },
        "cnt_not_null_context_merchant": {
            "output_type": dtypes.int64
        },
        "cnt_not_null_recent_txn": {
            "output_type": dtypes.int64
        },
    }

    def transform_row(row_dict):
        # Trim all string tensors
        features_dict = { column:
                        (tf.strings.strip(tensor) if tensor.dtype == 'string' else tensor)
                        for (column, tensor) in row_dict.items()
                        }
        return features_dict

    def read_bigquery(table_name):
        tensorflow_io_bigquery_client = BigQueryClient()
        read_session = tensorflow_io_bigquery_client.read_session(
            "projects/" + project_id,
            project_id, table_name, dataset_id,
            SELECTED_FIELDS_SCHEMA,
            requested_streams=2)

        dataset = read_session.parallel_read_rows()
        transformed_ds = dataset.map(transform_row)
        return transformed_ds

    logger.info(f"Downloading data from BigQuery resource: {project_id}.{dataset_id}.{table_id}...")
    ds = read_bigquery(table_id).shuffle(shuffle_buffer_size, reshuffle_each_iteration=False)

    return ds

def load_training_data(
        training_cfg: DictConfig,
        part: str = 'train',
        download: bool = False,
        last_checkpoint_date: str = "",
        random_seed: int = 42
    ):
    """Load the training data

    Args:
        training_cfg (DictConfig): the project config defined using hydra package
        part (str, optional): specify which part of the training data to load. Can be {'train', 'validation', 'test', 'full'}. Defaults to 'train'.
            where 'full' means download all the latest data, ready for training model for serving on production
        download (bool, optional): whether to download the data from remote source. Defaults to False.
        last_checkpoint_date (str, optional): only load input data after this checkpoint date (for incremental training)
        random_seed (int, optional): a seed to make sure the process can be reproduced. Defaults to 42.
        shuffle_buffer_size (int, optional): the size of the batch of records that gets shuffled one time. Defaults to 1M.

    Returns:
        Tensorflow Dataset: the requested dataset
    """
    local_rel_path = training_cfg.local_path
    local_root_path = os.path.abspath(os.path.join(__file__, '../../../'))
    local_abs_path = os.path.abspath(os.path.join(local_root_path, local_rel_path))
    local_part_path = os.path.join(local_abs_path, part)
    if download or not os.path.exists(local_part_path):
        logger.info("Triggering materialization from data_science_dbt...")
        update_materialized_table(selector="marts.recsys.dl_context.build", last_checkpoint_date=last_checkpoint_date, country_code=training_cfg.applied_country_code)
        logger.info("Downloading training data from BigQuery...")
        # If no local data then download all train, validation, test parts
        dataset_id = training_cfg.bigquery.dataset_id
        train_table_id = training_cfg.bigquery.train_table_id
        validation_table_id = training_cfg.bigquery.validation_table_id
        test_table_id = training_cfg.bigquery.test_table_id
        full_table_id = training_cfg.bigquery.full_table_id
        project_id = training_cfg.bigquery.project_id

        tf.random.set_seed(random_seed)
        train_ds = download_bigquery_table(project_id, dataset_id, train_table_id)
        validation_ds = download_bigquery_table(project_id, dataset_id, validation_table_id)
        test_ds = download_bigquery_table(project_id, dataset_id, test_table_id)
        full_ds = download_bigquery_table(project_id, dataset_id, full_table_id)

        # Persist
        tf.data.experimental.save(train_ds, os.path.join(local_abs_path, 'train'))
        tf.data.experimental.save(validation_ds, os.path.join(local_abs_path, 'validation'))
        tf.data.experimental.save(test_ds, os.path.join(local_abs_path, 'test'))
        tf.data.experimental.save(full_ds, os.path.join(local_abs_path, 'full'))

    logger.info(f"Loading {part} data from disk {local_part_path}...")
    ds = tf.data.experimental.load(local_part_path)
    return ds


def prep_training_data(ds, conf_input_prep):
    """ Prep training data into trainable format

    Args:
        ds (Tensorflow Dataset): the raw input dataset
        conf_input_prep (ConfigDict): the config for the input prep process,
            containing things such as event weights

    Returns:
        Tensorflow Dataset
    """
    DEFAULT_EVENT_WEIGHT = 1

    def get_weight(event_name: str, default_weight: int = 1) -> int:
        """ Because there will be error if we directly pass the dictionary to the
            map function of a Tensorflow Dataset, we define the function in advance

        Args:
            event_name (str): event name, e.g. View_Merchant
            default_weight (int): default value for weight in case of missing

        Returns:
            int: the mapped weight value
        """
        # TODO: Ideally we should get the weight values from config map where user
        # can change easily. But I haven't found a way to do this in Tensorflow so
        # there's hard code here.
        if event_name == 'View_Merchant':
            return default_weight
        if event_name == 'Transaction_Success':
            return 5
        return default_weight

    return ds.map(lambda x: {
        "target_merchant_id": x["merchant_id"],
        "context_search_terms": tf.strings.split(x['prev_search_term_list'], sep='|'),
        "context_merchants": tf.strings.split(x['prev_event_ruid_list'], sep='|'),
        "recent_txn_merchants": tf.strings.split(x['recent_txn_ruid_list'], sep='|'),
        "context_search_terms_time_recency": tf.strings.split(x['prev_search_term_time_diff_seconds_list'], sep='|'),
        "context_merchants_time_recency": tf.strings.split(x['prev_event_time_diff_seconds_list'], sep='|'),
        "recent_txn_time_recency": tf.strings.split(x['recent_txn_time_diff_days_list'], sep='|'),
        "cnt_not_null_context_search_term": x["cnt_not_null_context_search_term"],
        "cnt_not_null_context_merchant": x["cnt_not_null_context_merchant"],
        "cnt_not_null_recent_txn": x["cnt_not_null_recent_txn"],
        "user_id": x["user_id"],
        "sample_weight": get_weight(x['event_name'], DEFAULT_EVENT_WEIGHT)
    })


def get_vocab(ds, field_name: str) -> np.ndarray:
    """ Collect all the unique values for an attribute in the MapDataset

    Args:
        ds (Tensorflow Dataset): The dataset that containing the values
        field_name (str): the name of the attribute in the dataset to collect values

    Returns:
        Numpy Array: the concrete unique values
    """
    values = ds.batch(1_000_000).map(lambda x: x[field_name])
    unique_values = np.unique(np.concatenate(list(values)))
    return unique_values
