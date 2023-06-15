import urllib
from pathlib import Path

import yaml


def _is_url(path: str) -> bool:
    return bool(urllib.parse.urlparse(path).scheme)


def download_file(url: str, target_file: Path):
    """Download a file from a URL.

    Parameters
    ----------
    url : str
        URL to the file.
    target_file : Path
        Path to the target file.
    """
    urllib.request.urlretrieve(url, target_file)  # noqa: S310


def load_model_rdf(path: str | Path) -> dict:
    """Load a model from a RDF file.

    Parameters
    ----------
    path : str | Path
        Path to the RDF file.

    Returns
    -------
    model : dict
        Model as a dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def extract_urls(model: dict | list):
    """Extract urls from a model.

    Parameters
    ----------
    model : dict
        Model as a dictionary.

    Returns
    -------
    urls : list of strings
        list of url strings.
    """
    res = []
    if isinstance(model, dict):
        model = model.values()

    for value in model:
        if isinstance(value, (dict, list)):
            res.extend(extract_urls(value))
        elif isinstance(value, str) and _is_url(value):
            res.append(value)
    return res


def get_folder_name(model: dict):
    """
    Get the folder name for a model.
    """
    arch = model["weights"]["pytorch_state_dict"]["architecture_sha256"]
    weights = model["weights"]["pytorch_state_dict"]["sha256"]
    return f"{arch}_{weights}"


def import_model_from_rdf(path: str | Path, target_dir: Path):
    """Import a model from a RDF file.

    Parameters
    ----------
    path : str | Path
        Path to the RDF file.

    Returns
    -------
    model : dict
        Model as a dictionary.
    """
    model = load_model_rdf(path)

    for _key, value in model["weights"].items():
        if not _is_url(value["source"]):
            continue
        urllib.request.urlretrieve(  # noqa: S310
            value["source"], target_dir / value["source"].rsplit("/", 1)[-1]
        )
