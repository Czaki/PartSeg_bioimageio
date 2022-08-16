"""
Plugin specific settings
"""
import os

import appdirs
from pydantic import BaseSettings


class Settings(BaseSettings):
    save_model_dir: str = appdirs.user_data_dir("napari_biomodel")
    zenodo_token: str = ""

    def get_zenodo_token(self):
        return self.zenodo_token or os.environ.get("ZENODO_TOKEN", "")

    @property
    def model_summary_dict_path(self):
        return os.path.join(self.save_model_dir, "model_summary_dict.yaml")


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
