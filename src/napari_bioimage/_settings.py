"""
Plugin specific settings
"""
import os
from contextlib import suppress

import appdirs
from pydantic import BaseModel

try:
    from napari.settings import get_settings as naapari_get_settings

    def get_save_dir():
        return naapari_get_settings().config_path.parent

except ImportError:
    from pathlib import Path

    def get_save_dir():
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
        pass

        with open(get_save_dir() / SAVE_PATH, "w") as f:
            f.write(self.json(exclude={"zenodo_token"}))


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        settings_path = get_save_dir() / SAVE_PATH
        with suppress(Exception):
            if settings_path.exists():
                _settings = Settings.parse_file(settings_path)
                return _settings
        _settings = Settings()
    return _settings
