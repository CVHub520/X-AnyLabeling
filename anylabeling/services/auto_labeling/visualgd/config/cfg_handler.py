from . import cfg_fsc147
from ..util.slconfig import SLConfig


class ConfigurationHandler:
    @staticmethod
    def get_config(filename=None):
        if filename is None:
            return ConfigurationHandler.__get_default_cfg()
        else:
            return ConfigurationHandler.__get_cfg_from_file(filename)

    @staticmethod
    def __get_default_cfg():
        cfg_fsc147_dict = {
            k: v for k, v in vars(cfg_fsc147).items() if not k.startswith("__")
        }
        return SLConfig(cfg_dict=cfg_fsc147_dict)

    @staticmethod
    def __get_cfg_from_file(filename):
        return SLConfig.fromfile(filename)
