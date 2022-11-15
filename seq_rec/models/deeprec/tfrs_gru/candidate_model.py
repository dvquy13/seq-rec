import tensorflow as tf


class CandidateModel(tf.keras.Model):
    def __init__(self, model_hparams: dict):
        super().__init__()
        self.model_hparams = model_hparams

        target_input = tf.keras.Input(shape=[None,], dtype=tf.string)
        x = tf.keras.layers.StringLookup(vocabulary=model_hparams['merchant_vocab'], mask_token=None)(target_input)
        merchant_embedding_output = tf.keras.layers.Embedding(len(model_hparams['merchant_vocab']) + 1, model_hparams['embedding_dimension'])(x)
        self.merchant_embedding = tf.keras.Model([target_input], merchant_embedding_output, name='target_embedding')

    def call(self, merchant_ids):
        return self.merchant_embedding(merchant_ids)
