import tensorflow as tf
import numpy as np


class QueryModel(tf.keras.Model):
    def __init__(self, model_hparams: dict):
        super().__init__()
        self.model_hparams = model_hparams
        time_recency_buckets = np.array(list(map(str, range(model_hparams['time_recency_num_buckets']))))

        # Context Merchants
        context_merchants_inputs = tf.keras.Input(shape=[None,], dtype=tf.string)
        x = tf.keras.layers.StringLookup(vocabulary=model_hparams['merchant_vocab'], mask_token=None)(context_merchants_inputs)  # If specifying mask_token = 'NULL' then weird indices error occurs... Anyway we don't need to specify the mask_token since the NULL is left out already because of using fixed vocab
        merchant_embedding = tf.keras.layers.Embedding(input_dim=len(model_hparams['merchant_vocab']) + 1, output_dim=model_hparams['embedding_dimension'])(x)

        context_merchants_time_recency_inputs = tf.keras.Input(shape=[None,], dtype=tf.string)
        x = tf.keras.layers.StringLookup(vocabulary=time_recency_buckets, mask_token=None)(context_merchants_time_recency_inputs)
        merchant_recency_embedding = tf.keras.layers.Embedding(input_dim=len(time_recency_buckets) + 1, output_dim=model_hparams['embedding_dimension'])(x)

        merchant_features_embedding = tf.concat([merchant_embedding, merchant_recency_embedding], axis=2)
        context_merchants_outputs = tf.keras.layers.GRU(model_hparams['embedding_dimension'])(merchant_features_embedding)
        self.context_merchants_embedding = tf.keras.Model([context_merchants_inputs, context_merchants_time_recency_inputs], context_merchants_outputs, name='context_merchants_embedding')

        # Context Search Terms
        context_search_terms_inputs = tf.keras.Input(shape=[None,], dtype=tf.string)
        self.search_term_string_lookup_layer = tf.keras.layers.StringLookup(
            max_tokens=model_hparams['max_search_term_tokens'],
            mask_token='NULL'
        )
        self.search_term_string_lookup_layer.adapt(model_hparams['search_vocab'])
        x = self.search_term_string_lookup_layer(context_search_terms_inputs)
        search_term_embedding = tf.keras.layers.Embedding(input_dim=self.search_term_string_lookup_layer.vocabulary_size(), output_dim=model_hparams['embedding_dimension'])(x)

        context_search_terms_time_recency_inputs = tf.keras.Input(shape=[None,], dtype=tf.string)
        x = tf.keras.layers.StringLookup(vocabulary=time_recency_buckets, mask_token=None)(context_search_terms_time_recency_inputs)
        search_term_recency_embedding = tf.keras.layers.Embedding(input_dim=len(time_recency_buckets) + 1, output_dim=model_hparams['embedding_dimension'])(x)

        search_term_features_embedding = tf.concat([search_term_embedding, search_term_recency_embedding], axis=2)
        context_search_terms_outputs = tf.keras.layers.GRU(model_hparams['embedding_dimension'])(search_term_features_embedding)
        self.context_search_terms_embedding = tf.keras.Model([context_search_terms_inputs, context_search_terms_time_recency_inputs], context_search_terms_outputs, name='context_search_terms_embedding')

        # Recent Transactions
        recent_txn_inputs = tf.keras.Input(shape=[None,], dtype=tf.string)
        x = tf.keras.layers.StringLookup(vocabulary=model_hparams['merchant_vocab'], mask_token=None)(recent_txn_inputs)  # If specifying mask_token = 'NULL' then weird indices error occurs... Anyway we don't need to specify the mask_token since the NULL is left out already because of using fixed vocab
        merchant_embedding = tf.keras.layers.Embedding(input_dim=len(model_hparams['merchant_vocab']) + 1, output_dim=model_hparams['embedding_dimension'])(x)

        recent_txn_time_recency_inputs = tf.keras.Input(shape=[None,], dtype=tf.string)
        x = tf.keras.layers.StringLookup(vocabulary=time_recency_buckets, mask_token=None)(recent_txn_time_recency_inputs)
        merchant_recency_embedding = tf.keras.layers.Embedding(input_dim=len(time_recency_buckets) + 1, output_dim=model_hparams['embedding_dimension'])(x)

        merchant_features_embedding = tf.concat([merchant_embedding, merchant_recency_embedding], axis=2)
        recent_txn_outputs = tf.keras.layers.GRU(model_hparams['embedding_dimension'])(merchant_features_embedding)
        self.recent_txn_embedding = tf.keras.Model([recent_txn_inputs, recent_txn_time_recency_inputs], recent_txn_outputs, name='recent_txn_embedding')

        # Adding user_id introduces huge overfit. Need to know how to control this overfit before adding this.
        # user_id_input = tf.keras.Input(shape=[None,], dtype=tf.string)
        # x = tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None)(user_id_input)
        # user_id_output = tf.keras.layers.Embedding(input_dim=len(unique_user_ids) + 1, output_dim=model_hparams['embedding_dimension'])(x)
        # self.user_embedding = tf.keras.Model(user_id_input, user_id_output, name='user_id_embedding')

    def call(self, inputs):
        return tf.concat([
            self.context_search_terms_embedding([inputs['context_search_terms'], inputs['context_search_terms_time_recency']]),
            self.context_merchants_embedding([inputs['context_merchants'], inputs['context_merchants_time_recency']]),
            self.recent_txn_embedding([inputs['recent_txn_merchants'], inputs['recent_txn_time_recency']]),
            # self.user_embedding(inputs['user_id'])
        ], axis=1)
