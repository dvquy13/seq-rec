from kfp.v2.dsl import (Input, Output, Dataset, component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def preprocess_op(
        raw_data_file: Input[Dataset],
        country_code: str,
        prep_ds_file: Output[Dataset],
        merchant_vocab_file: Output[Dataset],
        search_vocab_file: Output[Dataset]
    ):
    import tensorflow as tf
    import seq_rec.utils as utils
    import seq_rec.io as io

    cfg = utils.load_cfg()

    raw_ds = tf.data.experimental.load(raw_data_file.path)
    prep_ds = io.prep_training_data(raw_ds, cfg.input_prep)

    tf.data.experimental.save(prep_ds, prep_ds_file.path)

    merchant_vocab = io.get_vocab(raw_ds, 'merchant_id')
    search_vocab = io.get_vocab(prep_ds, 'context_search_terms')

    import cloudpickle
    merchant_vocab_file.metadata['country_code'] = country_code
    with open(merchant_vocab_file.path, "wb") as f:
        cloudpickle.dump(merchant_vocab, f)

    search_vocab_file.metadata['country_code'] = country_code
    with open(search_vocab_file.path, "wb") as f:
        cloudpickle.dump(search_vocab, f)
