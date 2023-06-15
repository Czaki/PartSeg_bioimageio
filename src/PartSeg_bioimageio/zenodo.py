from pathlib import Path

import requests
import yaml

from ._settings import get_settings

MODEL_SUMMARY_FILE_NAME = "model_summary_dict.yaml"


def get_data_from_zenodo(url, **kwargs):
    res = requests.get(
        url,
        params={"access_token": get_settings().get_zenodo_token(), **kwargs},
        timeout=10,
    )
    return res.json()


def get_all_models_gen():
    data = get_data_from_zenodo(
        "https://zenodo.org/api/records/", q='keywords:"bioimage.io:model"'
    )
    res = data["hits"]["hits"]
    while "next" in data["links"]:
        data = get_data_from_zenodo(data["links"]["next"])
        res += data["hits"]["hits"]
        yield
    return res


def get_all_models():
    model_gen = get_all_models_gen()
    try:
        while True:
            next(model_gen)
    except StopIteration as e:
        return e.value


def get_model_rdf(model_zenodo_dkt: dict):
    try:
        bucket = next(
            filter(lambda x: x["key"] == "rdf.yaml", model_zenodo_dkt["files"])
        )
    except StopIteration as e:
        raise ValueError("Model RDF not found.") from e
    url = bucket["links"]["self"]
    res = requests.get(url, timeout=10)
    if res.status_code != 200:
        raise ValueError("Model RDF not found.")
    return yaml.safe_load(res.text)


def get_ilastik_models(data):
    res = []
    for d in data:
        for ident in d["metadata"]["related_identifiers"]:
            if (
                ident["identifier"]
                == "https://bioimage.io/#/r/ilastik%2Filastik"
            ):
                res.append(d)
                break
    return res


def _get_file_url(model_data: dict, file_name: str) -> str:
    for d in model_data["files"]:
        if d["key"] == file_name:
            return d["links"]["self"]
    raise ValueError("Model RDF not found.")


def get_bucket_name(data_model: dict):
    return data_model["links"]["bucket"].rsplit("/", 1)[-1]


def _download_single_model_base(data_model: dict):
    bucket_name = get_bucket_name(data_model)
    save_dir = Path(get_settings().save_model_dir) / bucket_name
    if (save_dir / "rdf.yaml").exists():
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(_get_file_url(data_model, "rdf.yaml"), timeout=10)
    if response.status_code != 200:
        raise ValueError("Modle RDF download fail")
    model = yaml.safe_load(response.text)
    with open(save_dir / "rdf.yaml", "w") as f:
        f.write(response.text)
    if "covers" in model:
        for cover in model["covers"]:
            res_cover = requests.get(
                _get_file_url(data_model, cover), timeout=10
            )
            if res_cover.status_code != 200:
                print(
                    "failed to download cover",
                    _get_file_url(data_model, cover),
                )
                continue
            with open(save_dir / cover, "wb") as f:
                f.write(res_cover.content)
    if "thumb250" in data_model["links"]:
        response_img = requests.get(
            data_model["links"]["thumb250"], timeout=10
        )
        if response_img.status_code != 200:
            return
        with open(save_dir / "thumb.png", "wb") as f:
            f.write(response_img.content)


def download_basic_data(data: dict):
    errors = []

    for d in data:
        try:
            _download_single_model_base(d)
        except ValueError as e:
            errors.append(e)
        yield


def download_model_data(data_model: dict):
    bucket_name = get_bucket_name(data_model)
    save_dir = Path(get_settings().save_model_dir) / bucket_name

    for file_info in data_model["files"]:
        if (save_dir / file_info["key"]).exists():
            yield
            continue
        with requests.get(
            file_info["links"]["self"], stream=True, timeout=10
        ) as res:
            res.raise_for_status()
            with open(save_dir / file_info["key"], "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
        yield
