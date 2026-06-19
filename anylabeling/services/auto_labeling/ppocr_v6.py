import os
import tempfile

import yaml

from .ppocr_v5 import PPOCRv5


class PPOCRv6(PPOCRv5):
    """PaddlePaddle OCR-v6"""

    _TEMP_DICT_FILES = []
    _DET_DB_CONFIG_KEYS = [
        "det_db_thresh",
        "det_db_box_thresh",
        "det_db_unclip_ratio",
        "det_db_max_candidates",
    ]

    @staticmethod
    def _get_model_dir(config, model_path_key):
        model_path = config.get(model_path_key)
        if not model_path:
            return None

        model_abs_path = os.path.abspath(model_path)
        if os.path.exists(model_abs_path):
            return os.path.dirname(model_abs_path)

        config_file_path = config.get("config_file")
        if config_file_path:
            config_folder = os.path.dirname(config_file_path)
            model_abs_path = os.path.abspath(
                os.path.join(config_folder, model_path)
            )
            if os.path.exists(model_abs_path):
                return os.path.dirname(model_abs_path)

        return None

    @staticmethod
    def _load_model_yml(config, model_path_key):
        model_dir = PPOCRv6._get_model_dir(config, model_path_key)
        if not model_dir:
            return {}

        inference_yml = os.path.join(model_dir, "inference.yml")
        if not os.path.isfile(inference_yml):
            return {}

        with open(inference_yml, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _write_temp_rec_dict(characters):
        dict_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix="_ppocrv6_keys.txt",
            delete=False,
        )
        try:
            dict_file.write("\n".join(characters))
            dict_file.write("\n")
            return dict_file.name
        finally:
            dict_file.close()

    @staticmethod
    def get_rec_char_dict_path(config, current_dir):
        rec_char_dict_path = config.get("rec_char_dict_path")
        if rec_char_dict_path:
            configured_path = PPOCRv5.get_rec_char_dict_path(
                config, current_dir
            )
            if os.path.exists(configured_path):
                return configured_path

            package_path = os.path.join(
                current_dir, "configs", "ppocr", rec_char_dict_path
            )
            if os.path.exists(package_path):
                return package_path

            return configured_path

        model_config = PPOCRv6._load_model_yml(config, "rec_model_path")
        characters = (
            model_config.get("PostProcess", {}).get("character_dict") or []
        )
        if characters:
            dict_path = PPOCRv6._write_temp_rec_dict(characters)
            PPOCRv6._TEMP_DICT_FILES.append(dict_path)
            return dict_path

        return PPOCRv5.get_rec_char_dict_path(config, current_dir)

    @staticmethod
    def get_det_db_params(config):
        postprocess_config = PPOCRv6._load_model_yml(
            config, "det_model_path"
        ).get("PostProcess", {})
        params = {}
        if "thresh" in postprocess_config:
            params["det_db_thresh"] = postprocess_config["thresh"]
        if "box_thresh" in postprocess_config:
            params["det_db_box_thresh"] = postprocess_config["box_thresh"]
        if "unclip_ratio" in postprocess_config:
            params["det_db_unclip_ratio"] = postprocess_config["unclip_ratio"]
        if "max_candidates" in postprocess_config:
            params["det_db_max_candidates"] = postprocess_config[
                "max_candidates"
            ]
        for key in PPOCRv6._DET_DB_CONFIG_KEYS:
            if key in config:
                params[key] = config[key]
        return params

    def parse_args(self):
        args = super().parse_args()
        for key, value in self.get_det_db_params(self.config).items():
            setattr(args, key, value)
        return args
