from pathlib import Path
from typing import Callable

from magicgui.types import PathLike
from PartSegCore.segmentation import ROIExtractionResult
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    RestartableAlgorithm,
)
from PartSegCore.utils import BaseModel


class MulticlassROIExtractionParameters(BaseModel):
    base_threshold: float = 0.5
    model_path: PathLike = Path.home()


class BioimageioROIExtraction(RestartableAlgorithm):
    __argument_class__ = MulticlassROIExtractionParameters

    def calculation_run(
        self, report_fun: Callable[[str, int], None]
    ) -> ROIExtractionResult:
        pass

    @classmethod
    def get_name(cls) -> str:
        return "BioimageIO ROI Extraction"
