import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

import bioimageio.core


def load_model_from_rdf_file(file_path: str):
    """
    Load a model from a RDF dictionary.
    """
    return bioimageio.core.load_resource_description(file_path)


def load_model_from_rdf(rdf: dict, dirpath: str):
    import torch

    pytorch_dict = rdf["weights"]["pytorch_state_dict"]
    architecture_url, class_name = pytorch_dict["architecture"].rsplit(":", 1)
    architecture_file_name = architecture_url.rsplit("/", 1)[-1]
    module_name = (
        f"napari_biomodels.model.m{pytorch_dict['architecture_sha256']}"
    )
    if module_name in sys.modules:
        model_module = sys.modules[module_name]
    else:
        model_spec = spec_from_file_location(
            module_name, os.path.join(dirpath, architecture_file_name)
        )
        model_module = module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
    model_instance = getattr(model_module, class_name)(
        **pytorch_dict["kwargs"]
    )
    model_instance.load_state_dict(
        torch.load(os.path.join(dirpath, pytorch_dict["source"]))
    )
    model_instance.eval()
    return model_instance
