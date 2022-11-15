from kfp.v2.dsl import (Input, Output, Dataset, Artifact, Model, component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def model_op(
        train_prep_ds_file: Input[Dataset],
        test_prep_ds_file: Input[Dataset],
        merchant_vocab_file: Input[Dataset],
        search_vocab_file: Input[Dataset],
        checkpoint_dir: Input[Artifact],
        resume_training: bool,
        country_code: str,
        model_output_dir: Output[Model],
        cfg_overrides: list = [],
        use_val_ds: bool = True,
    ):
    import cloudpickle
    import tensorflow as tf

    import seq_rec.utils as utils
    from seq_rec.models.deeprec.tfrs_gru import TFRSGRUModel as Model

    model_hparams_folder = 'models/deeprec/conf'
    model_codename = 'tfrs_gru'
    model_version = "v01"
    random_seed = 13

    with open(merchant_vocab_file.path, "rb") as f:
        merchant_vocab = cloudpickle.load(f)
    with open(search_vocab_file.path, "rb") as f:
        search_vocab = cloudpickle.load(f)

    model_hparams = {
        **utils.load_cfg(
            model_hparams_folder,
            is_relative_from_root_path=True,
            overrides=[
                f"tfrs_gru={model_version}",
                *cfg_overrides
            ]
        )[model_codename],
        "merchant_vocab": merchant_vocab,
        "search_vocab": search_vocab,
        "random_seed": random_seed
    }

    print("model_hparams:")
    print(model_hparams)

    model = Model(model_hparams)
    if resume_training:
        print(f"Resume Training is enabled. Loading model weights from {checkpoint_dir.path}")
        model.load_weights(checkpoint_dir.path)

    print("Loading input data...")
    train_prep_ds = tf.data.experimental.load(train_prep_ds_file.path)
    if use_val_ds:
        test_prep_ds = tf.data.experimental.load(test_prep_ds_file.path)
    else:
        test_prep_ds = None

    model.fit(train_prep_ds, test_prep_ds, checkpoint_filepath=checkpoint_dir.path)

    # Export model
    model_output_dir.metadata['country_code'] = country_code
    model.export(model_output_dir.path)
