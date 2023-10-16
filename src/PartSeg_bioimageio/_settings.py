"""
Plugin specific settings
"""
import os
from contextlib import suppress

import appdirs
from pydantic import BaseModel


def get_save_dir():
    try:
        from napari.settings import get_settings as napari_get_settings

        return napari_get_settings().config_path.parent
    except ImportError:
        from pathlib import Path

        return Path(appdirs.user_data_dir("napari-bioimage"))


SAVE_PATH = "napari-bioimage.json"


class Settings(BaseModel):
    save_model_dir: str = appdirs.user_data_dir("napari_biomodel")
    last_model_dir: str = appdirs.user_data_dir("napari_biomodel")
    zenodo_token: str = ""

    def get_zenodo_token(self):
        return self.zenodo_token or os.environ.get("ZENODO_TOKEN", "")

    @property
    def model_summary_dict_path(self):
        return os.path.join(self.save_model_dir, "model_summary_dict.yaml")

    def dump(self):
        with open(get_save_dir() / SAVE_PATH, "w") as f:
            f.write(self.json(exclude={"zenodo_token"}))


_settings = None


def get_settings() -> Settings:
    global _settings  # noqa: PLW0603
    if _settings is None:
        settings_path = get_save_dir() / SAVE_PATH
        with suppress(Exception):
            if settings_path.exists():
                _settings = Settings.parse_file(settings_path)
                return _settings
        _settings = Settings()
    return _settings
