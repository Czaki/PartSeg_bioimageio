"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import itertools
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import bioimageio.core as bc
import imageio.v2 as imageio
import numpy as np
import xarray as xr
import yaml
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from magicgui.widgets import create_widget
from napari.layers import Image
from napari.qt import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage import measure

from ._settings import get_settings
from .zenodo import (
    MODEL_SUMMARY_FILE_NAME,
    download_basic_data,
    download_model_data,
    get_all_models_gen,
    get_bucket_name,
)

if TYPE_CHECKING:
    import napari


def refresh_models_remote():
    gen = get_all_models_gen()
    try:
        while True:
            next(gen)
            yield
    except StopIteration as e:
        with open(
            os.path.join(
                get_settings().save_model_dir, MODEL_SUMMARY_FILE_NAME
            ),
            "w",
        ) as f:
            yaml.dump(e.value, f)
        yield from download_basic_data(e.value)


def _rdf_path(model_data, directory_path):
    return Path(directory_path) / get_bucket_name(model_data) / "rdf.yaml"


def load_model_data(directory_path: str, all_model_data: list):
    """
    Loads the model data from the given directory.
    """
    model_data_li = []
    directory_path = Path(directory_path)

    for model_data in all_model_data:
        rdf_path = _rdf_path(model_data, directory_path)
        if not rdf_path.exists():
            yield
            continue
        with rdf_path.open() as f:
            rdf_data = yaml.safe_load(f.read())
        cover = None
        if (
            len(rdf_data["covers"]) > 0
            and (
                cover_path := rdf_path.parent / rdf_data["covers"][0]
            ).exists()
        ):
            cover = imageio.imread(cover_path)
        model_data_li.append(
            {"rdf": rdf_data, "cover": cover, "model_data": model_data}
        )

        yield

    return model_data_li


def _load_bioimage_model(file_path):
    return bc.load_resource_description(file_path)


class NumpyQImage(QImage):
    """
    Class for fix problem with PySide2 QImage implementation
    (non copied buffer)
    """

    def __init__(self, image: np.ndarray):
        if image.ndim == 2:
            super().__init__(
                image.data,
                image.shape[1],
                image.shape[0],
                QImage.Format.Format_Grayscale8,
            )
        else:
            super().__init__(
                image.data,
                image.shape[1],
                image.shape[0],
                image.dtype.itemsize * image.shape[1] * image.shape[2],
                QImage.Format.Format_RGBA8888
                if image.shape[2] == 4
                else QImage.Format.Format_RGB888,
            )
        self.image = image


class ModelListElWidget(QFrame):
    def __init__(
        self,
        rdf: dict,
        cover: Union[str, QImage, np.ndarray],
        model_data: dict,
        load_model_func: callable,
        parent: Optional["QWidget"] = None,
        flags=Qt.WindowFlags(),
    ):
        super().__init__(parent, flags)
        self.name = QLabel(f"<b>{rdf['name']}</b>")
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name.setWordWrap(True)
        self.description = QLabel(rdf["description"])
        self.description.setWordWrap(True)
        if isinstance(cover, str):
            cover = QImage(cover)
        if isinstance(cover, np.ndarray):
            cover = NumpyQImage(cover)
        self.image = QLabel()
        if cover is not None:
            self.image.setPixmap(QPixmap.fromImage(cover.scaledToHeight(100)))
        self.btn = QPushButton("Load")
        self.btn.clicked.connect(self.btn_clicked)

        self.model_data = model_data
        self.rdf = rdf
        self.load_model_func = load_model_func

        layout = QVBoxLayout(self)
        layout.addWidget(self.name)
        layout.addWidget(self.description)
        layout.addWidget(self.image)
        layout.addWidget(self.btn)

        self.setFrameShadow(QFrame.Shadow.Sunken)

    def btn_clicked(self):
        self.load_model_func(self.rdf, self.model_data)


class ModelListView(QListWidget):
    pass


class ModelListWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.model_list = ModelListView(self)

        self.open_dir_with_data_btn = QPushButton("Open dir with data")
        self.open_dir_with_data_btn.clicked.connect(self._open_data_dir)
        self.refresh_btn = QPushButton("Refresh list of models")
        self.refresh_btn.clicked.connect(self.refresh_models)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.model_list)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.open_dir_with_data_btn)

        self.setLayout(layout)

        self.load_models()

    def load_models(self):
        num = 0

        def _inc():
            nonlocal num
            num += 1
            self.progress_bar.setValue(num)

        summary_path = (
            Path(get_settings().save_model_dir) / MODEL_SUMMARY_FILE_NAME
        )
        if not summary_path.exists():
            self.refresh_models()
            return

        with summary_path.open("r") as f:
            all_model_data = yaml.safe_load(f)

        _load_model_data = thread_worker(
            load_model_data,
            connect={
                "finished": self._hide_progress,
                "yielded": _inc,
                "returned": self._set_models,
            },
        )

        self.progress_bar.setRange(0, len(all_model_data))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        _load_model_data(get_settings().save_model_dir, all_model_data)

    def _hide_progress(self) -> None:
        self.progress_bar.setVisible(False)

    def _set_models(self, data: list):
        self.model_list.clear()
        for el in data:
            item = QListWidgetItem(el["rdf"]["name"], self.model_list)
            self.model_list.addItem(item)
            widget = ModelListElWidget(load_model_func=self.load_model, **el)
            item.widget = widget
            item.setSizeHint(widget.sizeHint())
            self.model_list.setItemWidget(item, widget)

    def refresh_models(self):

        _refresh_models = thread_worker(
            refresh_models_remote, connect={"finished": self.load_models}
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        _refresh_models()

    def load_model(self, model_rdf, model_zenodo):
        num = 0

        def _inc():
            nonlocal num
            num += 1
            self.progress_bar.setValue(num)

        _load = thread_worker(
            download_model_data,
            connect={
                "finished": self._hide_progress,
                "yielded": _inc,
                "returned": partial(self._load_model, model_rdf, model_zenodo),
            },
        )
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(model_zenodo["files"]))
        _load(model_zenodo)

    def _load_model(self, model_rdf, model_zenodo):
        widget: ModelWidget = self.viewer.window.add_plugin_dock_widget(
            "napari-bioimage", "Model control"
        )[1]
        widget.load_rdf(
            model_rdf,
            _rdf_path(model_zenodo, get_settings().save_model_dir),
        )

    def _open_data_dir(self):
        if sys.platform in ["linux", "linux2"]:
            subprocess.Popen(
                ["xdg-open", get_settings().save_model_dir]
            )  # nosec
        elif sys.platform == "darwin":
            subprocess.Popen(["open", get_settings().save_model_dir])  # nosec
        elif sys.platform == "win32":
            os.startfile(get_settings().save_model_dir)  # nosec


class ModelWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.model_rdf = None
        self.data_path = ""
        self.model_data = None

        self.header = QLabel()
        self.header.setAlignment(Qt.AlignCenter)
        self.load_model_from_disc_btn = QPushButton("Load model from disc")
        self.load_model_from_disc_btn.clicked.connect(
            self.load_model_from_disc
        )
        self.image_layer = create_widget(
            annotation=Image, label="Image", options={}
        )
        self.image_layer2 = create_widget(
            annotation=Image, label="Image", options={}
        )

        self.image_layer3 = create_widget(
            annotation=Image, label="Image", options={}
        )

        self.run_model_btn = QPushButton("Run model")
        self.run_model_btn.clicked.connect(self.run_model)
        self.run_model_btn.setEnabled(False)
        self.load_sample_data_btn = QPushButton("Sample data")
        self.load_sample_data_btn.setVisible(False)
        self.load_sample_data_btn.clicked.connect(self._load_sample_data)

        self.cover = QLabel()

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.image_layer.native)
        layout.addWidget(self.image_layer2.native)
        layout.addWidget(self.image_layer3.native)
        layout.addWidget(self.run_model_btn)
        layout.addWidget(self.load_sample_data_btn)
        layout.addWidget(self.load_model_from_disc_btn)
        self.reset_choices()

    def load_rdf(self, model_rdf, rdf_path: str):

        bc.load_resource_description(rdf_path)
        self.model_rdf = model_rdf
        self.header.setText(f'<b>{model_rdf["name"]}</b>')
        self.load_sample_data_btn.setVisible(True)

        _load = thread_worker(
            _load_bioimage_model,
            connect={"returned": self._load_model_finished},
        )
        _load(rdf_path)

    def _load_model_finished(self, model_data):
        self.model_data = model_data
        self.run_model_btn.setEnabled(True)

    def run_model(self):
        pred_pipeline = create_prediction_pipeline(
            bioimageio_model=self.model_data
        )
        input_ = self.image_layer.value.data
        input2_ = self.image_layer2.value.data
        input3_ = self.image_layer2.value.data
        input_ = np.concatenate([input_, input2_, input3_], axis=-3)
        axes = tuple(self.model_data.inputs[0].axes)
        if len(axes) > input_.ndim:
            input_ = input_.reshape(
                (1,) * (len(axes) - input_.ndim) + input_.shape
            )
        input_tensor = xr.DataArray(input_, dims=axes)

        prediction = pred_pipeline(input_tensor)[0]
        self.viewer.add_image(
            prediction, name="Prediction", scale=self.image_layer.value.scale
        )

    def _load_sample_data(self):
        if self.model_data is None:
            return
        print(self.model_data.test_inputs, self.model_data.test_outputs)
        for filename in itertools.chain(
            self.model_data.test_inputs, self.model_data.test_outputs
        ):

            self.viewer.add_image(np.load(filename), name=filename.name)

    def load_model_from_disc(self):
        file_name, ok = QFileDialog.getOpenFileName(
            self,
            "Open file",
            get_settings().last_model_dir,
            "Model files (*.yaml)",
        )
        if not ok:
            return
        get_settings().last_model_dir = os.path.dirname(file_name)
        get_settings().dump()
        with open(file_name) as f:
            model_rdf = yaml.safe_load(f)
        self.load_rdf(model_rdf, file_name)

    def reset_choices(self, event=None):
        self.image_layer.reset_choices(event)
        self.image_layer2.reset_choices(event)
        self.image_layer3.reset_choices(event)

    def showEvent(self, event) -> None:
        self.reset_choices()
        return super().showEvent(event)


class ReconstructMultipleClassFromLayer(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.class_layer = create_widget(
            annotation=Image, label="Class", options={}
        )
        self.data_threshold = QDoubleSpinBox(self)
        self.data_threshold.setValue(0.5)
        self.reconstruct_btn = QPushButton("Reconstruct")
        self.reconstruct_btn.clicked.connect(self._reconstruct)

        layout = QVBoxLayout(self)
        layout.addWidget(self.class_layer.native)
        layout.addWidget(self.data_threshold)
        layout.addWidget(self.reconstruct_btn)

    def _reconstruct(self):
        thr_val = self.data_threshold.value()
        class_data = np.array(self.class_layer.value.data)

        max_class = class_data.argmax(axis=-3)
        max_class[class_data.max(axis=-3) < thr_val] = 0

        components = measure.label(max_class > 0, connectivity=1)
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

        self.viewer.add_labels(
            res,
            name="Reconstructed",
            scale=self.class_layer.value.scale[-res.ndim :],  # noqa: E203
        )
        # self.viewer.add_labels(max_class, name="Max class")
        # self.viewer.add_labels(components, name="Components")

    def reset_choices(self, event=None):
        self.class_layer.reset_choices(event)

    def showEvent(self, event) -> None:
        self.reset_choices()
        return super().showEvent(event)
