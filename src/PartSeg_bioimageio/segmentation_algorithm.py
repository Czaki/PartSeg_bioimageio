from typing import Callable

import numpy as np
import SimpleITK
import xarray as xr
from bioimageio.core import load_resource_description
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from PartSegCore.project_info import AdditionalLayerDescription
from PartSegCore.segmentation import ROIExtractionResult
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    RestartableAlgorithm,
)
from PartSegCore.utils import BaseModel
from pydantic import Field
from skimage import measure

from .partseg_widgets import BioImageModel


class MulticlassROIExtractionParameters(BaseModel):
    base_threshold: float = 0.5
    model_path: BioImageModel = Field(
        BioImageModel(path="", channels=[]), title="Model"
    )
    minimum_size: int = Field(20, title="Minimum size (px)")


class BioimageioROIExtraction(RestartableAlgorithm):
    __argument_class__ = MulticlassROIExtractionParameters

    def __init__(self):
        super().__init__()
        self.prediction_pipeline = None
        self.model_data = None
        self.prediction = None
        self.labeling = None
        self.components = None

    def calculation_run(
        self, report_fun: Callable[[str, int], None]
    ) -> ROIExtractionResult:
        restarted = False
        if (
            self.prediction_pipeline is None
            or self.new_parameters.model_path.path
            != self.parameters.get("model_path", BioImageModel("", ())).path
        ):
            self.model_data = load_resource_description(
                self.new_parameters["model_path"].path
            )
            self.prediction_pipeline = create_prediction_pipeline(
                bioimageio_model=self.model_data
            )
            restarted = True
        if restarted or (
            self.new_parameters.model_path.channels
            != self.parameters.get(
                "model_path", BioImageModel("", ())
            ).channels
        ):
            data = [
                self.image.get_channel(i)
                for i in self.new_parameters.model_path.channels
            ]
            self.parameters["model_path"] = self.new_parameters.model_path
            axes = tuple(self.model_data.inputs[0].axes)
            input_ = xr.DataArray(np.concatenate(data, axis=-3), dims=axes)
            self.prediction = np.array(self.prediction_pipeline(input_)[0])
            restarted = True
        if (
            restarted
            or self.new_parameters.base_threshold
            != self.parameters.get("base_threshold", None)
            or self.new_parameters.minimum_size
            != self.parameters.get("minimum_size", None)
        ):
            self._reconstruct()

        return ROIExtractionResult(
            roi=self.components,
            parameters=self.get_segmentation_profile(),
            additional_layers={
                "prediction": AdditionalLayerDescription(
                    self.prediction, "image", "prediction"
                )
            },
            alternative_representation={
                "Labeling": self.labeling,
            },
        )

    def _reconstruct(self):

        max_class = self.prediction.argmax(axis=-3)
        max_class[
            self.prediction.max(axis=-3) < self.new_parameters.base_threshold
        ] = 0

        components = SimpleITK.GetArrayFromImage(
            SimpleITK.RelabelComponent(
                SimpleITK.ConnectedComponent(
                    SimpleITK.GetImageFromArray(
                        (max_class > 0).astype(np.uint8)
                    )
                ),
                self.new_parameters.minimum_size,
            )
        )

        res = np.zeros(components.shape, dtype=np.uint16)
        props = measure.regionprops(components, max_class)

        bbox_step = len(props[0].bbox) // 2

        for prop in props:
            component_num = np.bincount(prop.image_intensity[prop.image])[
                1:
            ].argmax()
            bbox = tuple(
                slice(x, y)
                for x, y in zip(prop.bbox[:bbox_step], prop.bbox[bbox_step:])
            )
            res[bbox][prop.image] = component_num + 1
        self.labeling = res
        self.components = components

    @classmethod
    def get_name(cls) -> str:
        return "BioimageIO Multilabel"

    def get_info_text(self):
        return f"Found {np.max(self.components)} components"
