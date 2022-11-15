import requests
import logging

logger = logging.getLogger(__name__)

def update_materialized_table(
        selector: str,
        last_checkpoint_date: str = "",
        country_code: list = ['SG']
    ):
    """ Update the materialized tables in BigQuery to get the latest training data.

    Args:
        selector (str): DBT build select patterns as we calling the dbt data_science_dbt
        last_checkpoint_date (str): ISO format, example: 2022-09-07
        country_code (list[str]): list of country code to get the data
    """
    # TODO: Fill in the Cloud Run service of data_science_dbt DBT
    base_url = "https://<DBT_CLOUD_RUN_URL>/build_selected"
    params = f"select={selector}&sync_datahub=0"
    var_params = []
    if last_checkpoint_date:
        param_str = 'dl_context__event_start_date: %s' % (last_checkpoint_date)
        var_params.append(param_str)
    if country_code:
        param_str = 'dl_context__applied_country_code: %s' % (country_code)
        var_params.append(param_str)
    if var_params:
        var_params_str = ','.join(var_params)
        params += "&vars={%s}" % (var_params_str)
    url = f"{base_url}?{params}"
    logger.debug(f"url: {url}")

    headers = {}

    response = requests.request(
        "GET",
        url,
        headers=headers,
        timeout=None  # Building the materialized tables for seq_rec take about 10 minutes
    )

    logger.info(response.text)

    if not response.ok:
        raise Exception("Build fails! Check above logs for more info.")
