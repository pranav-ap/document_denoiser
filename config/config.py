import hydra
from omegaconf import DictConfig


config: DictConfig | None = None


def setup_config():
    hydra.initialize(version_base=None, config_path=".")
    cfg = hydra.compose("config")

    global config
    config = cfg


setup_config()
