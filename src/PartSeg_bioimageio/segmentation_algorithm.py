from typing import Callable

from PartSegCore.segmentation import ROIExtractionResult
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    RestartableAlgorithm,
)
from PartSegCore.utils import BaseModel
from pydantic import Field

from .partseg_widgets import BioImageModel


class MulticlassROIExtractionParameters(BaseModel):
    base_threshold: float = 0.5
    model_path: BioImageModel = Field(
        BioImageModel(path="", channels=[]), title="Model"
    )
    val: int = 1


class BioimageioROIExtraction(RestartableAlgorithm):
    __argument_class__ = MulticlassROIExtractionParameters

    def calculation_run(
        self, report_fun: Callable[[str, int], None]
    ) -> ROIExtractionResult:
        pass

    @classmethod
    def get_name(cls) -> str:
        return "BioimageIO ROI Extraction"
