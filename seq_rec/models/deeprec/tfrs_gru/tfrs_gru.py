import os

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow_recommenders as tfrs
# To handle error not able to save model_hparams object
# from tensorflow.python.training.tracking.data_structures import NoDependency

from .query_model import QueryModel
from .candidate_model import CandidateModel

from seq_rec.metrics import metrics
from seq_rec.models.deeprec.ann_helper import ANNHelper

class TFRSGRUModel(tfrs.models.Model):
    def __init__(self, model_hparams: dict):
        self.model_hparams = model_hparams
        super().__init__()
        # self.model_hparams = NoDependency(model_hparams)
        logger.debug("Initializing Query Model...")
        self.query_model = tf.keras.Sequential([
            QueryModel(model_hparams),
            tf.keras.layers.Dense(model_hparams['embedding_dimension']),
        ])
        logger.debug("Initializing Candidate Model...")
        self.candidate_model = tf.keras.Sequential([
            CandidateModel(model_hparams),
            tf.keras.layers.Dense(model_hparams['embedding_dimension']),
        ])
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(model_hparams['merchant_vocab']).batch(128).map(self.candidate_model),
            ),
            # batch_metrics=[tf.keras.metrics.AUC]  # Can not use because missing y_pred error
        )

        # Compile
        logger.info("Compiling model...")
        optimizer = tf.keras.optimizers.Adam(self.model_hparams['learning_rate'])
        # self.model_hparams['optimizer'] = optimizer.get_config()
        super().compile(optimizer=optimizer)

        # Placeholder for the Approximate Nearest Neighbors index after model training
        self.index = None

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            "context_search_terms": features["context_search_terms"],
            "context_merchants": features["context_merchants"],
            "recent_txn_merchants": features["recent_txn_merchants"],
            "context_merchants_time_recency": features["context_merchants_time_recency"],
            "context_search_terms_time_recency": features["context_search_terms_time_recency"],
            "recent_txn_time_recency": features["recent_txn_time_recency"],
            # "user_id": features["user_id"]
        })
        candidate_embeddings = self.candidate_model(features['target_merchant_id'])
        sample_weight = features['sample_weight']

        return self.task(query_embeddings, candidate_embeddings, sample_weight)

    def fit(self, training_data, val_data=None, checkpoint_filepath: str = 'checkpoint/'):
        """ Compile and fit the model with training data

        Args:
            training_data (Tensorflow Dataset): training data
            val_data (Tensorflow Dataset): validation data
            checkpoint_filepath (str): path to save checkpoint weights for resuming model
    """
        logger.info("Caching training data...")
        cached_train = training_data.shuffle(1_000_000).batch(self.model_hparams['batch_size']).cache()
        cached_eval = None
        val_metric = "factorized_top_k/top_10_categorical_accuracy"
        if val_data is not None:
            cached_eval = val_data.batch(self.model_hparams['batch_size']).cache()
            val_metric = self.model_hparams.get('val_metric', val_metric)

        callback_early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=val_metric,
            min_delta=0.001,
            patience=3,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )

        callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor=val_metric,
            mode='max',
            save_best_only=True
        )

        logger.info("Fitting model...")
        super().fit(
            cached_train,
            epochs=self.model_hparams['epochs'],
            callbacks=[callback_early_stopping, callback_model_checkpoint],
            validation_data=cached_eval
        )

        # Build index for serving
        logger.info("Indexing trained model for serving...")
        self.index = tfrs.layers.factorized_top_k.BruteForce(self.query_model)
        self.index.index_from_dataset(
            tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensor_slices(self.model_hparams['merchant_vocab']).batch(100),
                    tf.data.Dataset.from_tensor_slices(self.model_hparams['merchant_vocab']).batch(100).map(self.candidate_model)
                )
            )
        )

    def run_eval(self, val_ds, k = 10) -> dict:
        """ Evaluate the predictions against a validation dataset

        Args:
            val_ds (Tensorflow Dataset): the validation dataset
            k (int): number of predictions returned

        Returns:
            dict: contains metric results
        """
        # Built-in evaluation from model
        cached_ds = val_ds.batch(self.model_hparams['batch_size']).cache()
        eval_result = super().evaluate(cached_ds, return_dict=True)

        # General evaluation
        if self.index is not None:
            labels = list(val_ds.map(lambda x: x['target_merchant_id']).as_numpy_iterator())
            val_ds_input = ANNHelper.prepare_batch_input(val_ds)
            rec_scores, rec_merchant_ids = self.index(val_ds_input, k=k)
            coverage = metrics.calc_coverage(rec_merchant_ids, catalog=self.model_hparams['merchant_vocab'])
            accuracy_measures = metrics.calc_accuracy(rec_merchant_ids, labels)
            eval_result = {
                **eval_result,
                **accuracy_measures,
                'coverage': coverage
            }

        return eval_result

    def export(self, path: str):
        """ Export the ANN index model for serving.

        Args:
            path (str): local path where the model artifacts are stored
        """
        signature_dict = {
            'context_merchants': tf.TensorSpec(shape=[None, None], dtype=tf.string, name='context_merchants'),
            'context_search_terms': tf.TensorSpec(shape=[None, None], dtype=tf.string, name='context_search_terms'),
            'recent_txn_merchants': tf.TensorSpec(shape=[None, None], dtype=tf.string, name='recent_txn_merchants'),
            'context_merchants_time_recency': tf.TensorSpec(shape=[None, None], dtype=tf.string, name='context_merchants_time_recency'),
            'context_search_terms_time_recency': tf.TensorSpec(shape=[None, None], dtype=tf.string, name='context_search_terms_time_recency'),
            'recent_txn_time_recency': tf.TensorSpec(shape=[None, None], dtype=tf.string, name='recent_txn_time_recency'),
        }

        @tf.function(input_signature=[signature_dict])
        def rec_at_10(data):
            result = self.index(data, k=10)
            return {
                "scores": result[0],
                "merchant_id": result[1]
            }

        @tf.function(input_signature=[signature_dict])
        def rec_at_100(data):
            result = self.index(data, k=100)
            return {
                "scores": result[0],
                "merchant_id": result[1]
            }

        tf.saved_model.save(
            self.index,
            path,
            signatures={
                "serving_default": rec_at_100,
                "k_10": rec_at_10,
                "k_100": rec_at_100,
            }
        )

    def export_embeddings(self, base_log_dir: str):
        """ Export the embeddings to local disk

        Args:
            base_log_dir (str): base local path where the embeddings are stored
        """
        logger.info("Saving search term embeddings...")
        search_term_weights = self.query_model.layers[0].context_search_terms_embedding.layers[4].get_weights()[0]
        search_term_keys = self.query_model.layers[0].context_search_terms_embedding.layers[2].get_vocabulary()
        self._save_embeddings(base_log_dir, search_term_keys, search_term_weights, name='search_terms')

        logger.info("Saving merchant embeddings...")
        merchant_weights = self.query_model.layers[0].context_merchants_embedding.layers[4].get_weights()[0]
        merchant_keys = self.query_model.layers[0].context_merchants_embedding.layers[2].get_vocabulary()
        self._save_embeddings(base_log_dir, merchant_keys, merchant_weights, name='merchants')

    @staticmethod
    def _save_embeddings(base_log_dir, embedding_keys, embedding_values, name):
        """ Utils function to save the embeddings to disk

        Args:
            base_log_dir (str): base local path where the embeddings are stored
            embedding_keys (array): names of the embeddings, like merchant_ids
            embedding_values (array): the actual embedding vectors
            name (str): the name of folder that store the embeddings

        Raises:
            Exception: This function assumes that the log_dir already exists so that
                it will raise exception if user trying to overwrite the log_dir. This
                forces the user to examine the already saved embeddings if any.

        Returns:
            bool: Return True for successful operation
        """
        log_dir = f'{base_log_dir}/{name}/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            raise Exception(f'{log_dir} already exists')

        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            for key in embedding_keys:
                f.write("{}\n".format(key))

        weights = tf.Variable(embedding_values)
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        return True
