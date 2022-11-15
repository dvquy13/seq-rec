from kfp.v2.dsl import (Input, Artifact, component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def copy_output_to_gcs_op(
        output_obj: Input[Artifact],
        source_bucket_name: str,
        destination_bucket_name: str,
        destination_blob_name: str,
        overwrite_if_exists: bool = True
    ):
    """ Copy the output artifact from a Kubeflow Pipeline components to destination GCS
    """
    import os
    import glob
    from google.cloud import storage

    output_uri = output_obj.path
    print("output_uri")
    print(output_uri)
    blob_name = output_uri.replace('/gcs', '')
    blob_name = blob_name.replace("/" + source_bucket_name + "/", '')
    print("blob_name")
    print(blob_name)

    def copy_blob(
        bucket_name, blob_name, destination_bucket_name, destination_blob_name, overwrite_if_exists: bool = True
    ):
        """Copies a blob from one bucket to another with a new name."""
        # bucket_name = "your-bucket-name"
        # blob_name = "your-object-name"
        # destination_bucket_name = "destination-bucket-name"
        # destination_blob_name = "destination-object-name"

        storage_client = storage.Client()

        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(blob_name)
        # Assume that if source_blob does not exist then it might be a directory instead of a file
        if blob_name.endswith('/') or (not source_blob.exists() and not blob_name.endswith('/')):
            def upload_local_directory_to_gcs(local_path, bucket_name, gcs_path):
                gcs_client = storage.Client()

                bucket = gcs_client.get_bucket(bucket_name)
                assert os.path.isdir(local_path)
                for local_file in glob.glob(local_path + '/**'):
                    if not os.path.isfile(local_file):
                        upload_local_directory_to_gcs(local_file, bucket, os.path.join(gcs_path, os.path.basename(local_file)))
                    else:
                        remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
                        blob = bucket.blob(remote_path)
                        blob.upload_from_filename(local_file)

            upload_local_directory_to_gcs(output_obj.path, destination_bucket_name, destination_blob_name)
        else:
            destination_bucket = storage_client.bucket(destination_bucket_name)
            destination_blob = destination_bucket.blob(destination_blob_name)
            if destination_blob.exists() and not overwrite_if_exists:
                print(f"Destination blob {destination_blob} already exists. Abort!")
                return False

            blob_copy = source_bucket.copy_blob(
                source_blob, destination_bucket, destination_blob_name
            )

        print(
            "Blob {} in bucket {} copied to blob {} in bucket {}.".format(
                source_blob.name,
                source_bucket.name,
                destination_blob_name,
                destination_bucket_name,
            )
        )

        return True

    copy_blob(source_bucket_name, blob_name, destination_bucket_name, destination_blob_name, overwrite_if_exists=overwrite_if_exists)
