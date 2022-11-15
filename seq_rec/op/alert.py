from kfp.v2.dsl import component

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def slack_noti_exit_op(
        webhook_url: str,
        message: list,
        run_name: str,
        job_status_file_name: str,
        bucket_url: str,
        folder: str,
        pipeline_stakeholders_slack_uids: dict
    ):
    import json
    import requests

    job_succeeded = False
    try:
        from google.cloud import storage
        gcs_client = storage.Client()

        bucket_name = bucket_url.replace("gs://", "")
        bucket = gcs_client.get_bucket(bucket_name)
        gcs_file_path = f'{folder}/{job_status_file_name}'
        blob = bucket.blob(gcs_file_path)

        if blob.exists():
            print(f"{gcs_file_path} exists")
            blob = bucket.get_blob(gcs_file_path)

        with blob.open(mode='r') as f:
            job_status_str = f.read()

        import re
        job_status = re.search(r"Run name (.+?) ", job_status_str)
        if job_status:
            if job_status.group(1) == run_name:
                job_succeeded = True
    except Exception as e:
        import traceback
        traceback.print_exc(e)

    status = ":x: FAIL" if not job_succeeded else ":white_check_mark: SUCCESS"
    status = "*Status:* " + status
    text = [
        status
    ]
    if not job_succeeded:
        text_stakeholders = (f"<@{slack_uid}>" for slack_uid in pipeline_stakeholders_slack_uids.values())
        text_stakeholders = ' '.join(text_stakeholders)
        text.append(f"*Stakeholders:* {text_stakeholders}")
    text = '\n'.join(text)
    status_comp = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": text
        }
    }

    message.append(status_comp)

    def slack_noti(message: list, webhook_url: str):
        headers = {"Content-type": "application/json"}

        data = json.dumps({"blocks": message})

        r = requests.post(webhook_url, headers=headers, data=data)

        return r
    slack_noti(message=message, webhook_url=webhook_url)


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def slack_noti_op(
        webhook_url: str,
        message: list,
    ):
    import json
    import requests


    def slack_noti(message: list, webhook_url: str):
        headers = {"Content-type": "application/json"}

        data = json.dumps({"blocks": message})

        r = requests.post(webhook_url, headers=headers, data=data)

        return r

    slack_noti(message=message, webhook_url=webhook_url)


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def record_job_status_op(
        run_name: str,
        job_status_file_name: str,
        bucket_name: str,
        folder: str,
    ):
    from google.cloud import storage
    from datetime import datetime

    gcs_client = storage.Client()

    bucket = gcs_client.get_bucket(bucket_name)
    gcs_file_path = f'{folder}/{job_status_file_name}'
    blob = bucket.blob(gcs_file_path)

    if blob.exists():
        print(f"{gcs_file_path} exists")
        blob = bucket.get_blob(gcs_file_path)

    with blob.open(mode='w') as f:
        message = f"""
        Run name {run_name} has succeeded at {datetime.utcnow()} UTC
        """
        f.write(message)


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def record_last_checkpoint_date_op(
        checkpoint_bucket: str,
        last_checkpoint_date_blob_name: str
    ):
    from google.cloud import storage
    from datetime import datetime

    gcs_client = storage.Client()

    bucket = gcs_client.get_bucket(checkpoint_bucket)
    blob = bucket.blob(last_checkpoint_date_blob_name)

    if blob.exists():
        print(f"{last_checkpoint_date_blob_name} exists")
        blob = bucket.get_blob(last_checkpoint_date_blob_name)

    with blob.open(mode='w') as f:
        message = datetime.now().strftime("%Y-%m-%d")
        f.write(message)
