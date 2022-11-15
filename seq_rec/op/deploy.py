from kfp.v2.dsl import (component)

import seq_rec.utils as utils

cfg = utils.load_cfg()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def deploy_model_to_gcp_endpoint_op(
        model_bucket_name: str,
        model_blob_name: str,
        model_name: str,
        model_version: str,
        country_code: str,
        endpoint_id: str = None,
        traffic_split: dict = {"0": 100},
        undeploy_zero_traffic_models: bool = False,
    ):
    """ Deploy model to GCP Vertex AI Endpoint for serving. Majority of configs are defined
        in the hydra config.

    Args:
        model_bucket_name (str): the GCS bucket name that contain the model artifacts (not including gs://)
        model_blob_name (str): the path to the model directory in that GCS bucket
        model_name (str)
        model_version (str)
        country_code (str)
        endpoint_id (str): id of the existing endpoint to deploy to
        traffic_split (dict): the traffic split instructions to giv the Vertex AI endpoint given new model
    """
    import google.cloud.aiplatform as aip
    import seq_rec.utils as utils
    from datetime import datetime

    cfg = utils.load_cfg()

    PROJECT_ID = cfg.env.gcp.project_id
    MODEL_NAME = f"{model_name}_{country_code}"

    ENDPOINT_LOCATION = cfg.env.gcp.endpoint.location
    ENDPOINT_NAME = f"{cfg.env.gcp.endpoint.name}_{country_code}"
    MODEL_VERSION = model_version
    MODEL_VERSION_NODOT = MODEL_VERSION.replace(".", "")
    MODEL_NAME_VERSION = f"{MODEL_NAME}_{MODEL_VERSION_NODOT}"
    PATH_TO_MODEL_ARTIFACT_DIRECTORY = f"gs://{model_bucket_name}/{model_blob_name}"
    CONTAINER_IMAGE_URI = cfg.env.gcp.endpoint.container_image_uri
    ENDPOINT_MACHINE_TYPE = cfg.env.gcp.endpoint.machine_type
    ENDPOINT_MIN_REPLICA_COUNT = cfg.env.gcp.endpoint.min_replica_count
    ENDPOINT_MAX_REPLICA_COUNT = cfg.env.gcp.endpoint.max_replica_count

    aip.init(project=PROJECT_ID, location=ENDPOINT_LOCATION)

    endpoint = None
    # Select the existing endpoint
    if endpoint_id is not None and endpoint_id != "":
        endpoint = [endpoint for endpoint in aip.Endpoint.list() if endpoint.name == endpoint_id]
        if endpoint:
            endpoint = endpoint[0]
    if endpoint is None:
        # Check if endpoint_name already there
        endpoint = [endpoint for endpoint in aip.Endpoint.list() if endpoint.display_name == ENDPOINT_NAME]
        if endpoint:
            endpoint = endpoint[0]
        else:
            # Create new endpoint
            endpoint = aip.Endpoint.create(
                display_name=ENDPOINT_NAME,
                project=PROJECT_ID,
                location=ENDPOINT_LOCATION,
            )
    print(f"endpoint id: {endpoint.name}")
    print(f"endpoint name: {endpoint.display_name}")

    current_minute_str = datetime.now().strftime("%Y%m%d%H%M")
    # Upload model to registry
    model = aip.Model.upload(
        display_name=f"{MODEL_NAME_VERSION}_{current_minute_str}",
        artifact_uri=PATH_TO_MODEL_ARTIFACT_DIRECTORY,
        serving_container_image_uri=CONTAINER_IMAGE_URI,
        sync=False
    )

    model.wait()

    TRAFFIC_SPLIT = traffic_split
    DEPLOY_GPU = False
    if DEPLOY_GPU:
        ACCELERATOR_COUNT = 1
    else:
        ACCELERATOR_COUNT = 0

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=MODEL_NAME_VERSION,
        traffic_split=TRAFFIC_SPLIT,
        machine_type=ENDPOINT_MACHINE_TYPE,
        accelerator_type=DEPLOY_GPU,
        accelerator_count=ACCELERATOR_COUNT,
        min_replica_count=ENDPOINT_MIN_REPLICA_COUNT,
        max_replica_count=ENDPOINT_MAX_REPLICA_COUNT,
    )

    if undeploy_zero_traffic_models:
        deployed_model_ids_in_use = set()
        for deployed_model_id, split in endpoint.traffic_split.items():
            if split > 0:
                deployed_model_ids_in_use.add(deployed_model_id)
        for deployed_model in endpoint.list_models():
            if deployed_model.id not in deployed_model_ids_in_use:
                print(f"Undeploying DeployedModel {deployed_model.id} from endpoint {endpoint.name}...")
                endpoint.undeploy(deployed_model.id)
                model_id = deployed_model.model
                model_to_delete = aip.Model(model_id)
                print(f"Deleting Model {model_id}...")
                model_to_delete.delete()


@component(base_image=cfg.env.pipeline.kubeflow.main_image_uri)
def update_user_recent_txn_in_recommend_api_op(
        api_key: str,
        env: str
    ):
    import requests

    # Pushing mechanism for the service API to update new model
    url = f"<EXAMPLE_RECOMMEND_SERVICE_API_UPDATE_ENDPOINT>"

    payload = {}
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, timeout=600)

    print(response.text)

    if not response.ok:
        raise Exception("Update fails!")
