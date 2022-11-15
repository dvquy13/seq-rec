from operator import is_
from omegaconf import DictConfig
import os
from hydra import initialize_config_dir, compose

from dotenv import load_dotenv


def load_cfg(conf_dir: str = None, is_relative_from_root_path: bool = False, **kwargs) -> DictConfig:
    """ Load the hydra config

    Args:
        conf_dir (str): the path to the hydra config directory, can be relative path
        is_relative_from_root_path (bool): flag to know whether the path is relative or not, where
            relative means starting from root folder
        kwargs: keyword arguments for the hydra compose function

    Returns:
        DictConfig: the loaded config
    """
    if not conf_dir:
        cur_dir = os.path.abspath(os.path.join(__file__, "../../"))
        conf_dir = f"{cur_dir}/conf"
    else:
        if is_relative_from_root_path:
            root_dir = os.path.abspath(os.path.join(__file__, "../../"))
            conf_dir = os.path.join(root_dir, conf_dir)
        else:
            conf_dir = os.path.abspath(conf_dir)
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose("config", **kwargs)

    # Load env vars
    load_dotenv()

    return cfg
