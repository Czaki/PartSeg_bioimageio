import os
from functools import lru_cache
from glob import glob
from typing import Tuple

import numpy as np
import torch
from PartSegCore.analysis.load_functions import LoadProject
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch_em.segmentation import get_data_loader, get_raw_transform
from torch_em.util import ensure_tensor_with_channels


@lru_cache(maxsize=1)
def load_data(file_path):
    return LoadProject.load([file_path])


class PartSegProjectDataset(Dataset):
    def __init__(
        self,
        project_paths,
        label_name,
        patch_shape,
        dtype=torch.float32,
        label_dtype=torch.float32,
        label_transform=None,
        raw_transform=None,
        mul_factor=16,
    ):
        self.project_paths = project_paths
        self.label_name = label_name
        self.patch_shape = patch_shape
        self.dtype = dtype
        self.label_dtype = label_dtype
        self._cache = {}
        self._ndim = 2
        self.label_transform = label_transform
        self.raw_transform = raw_transform
        self._mul_factor = mul_factor

    def __getitem__(self, index) -> T_co:
        index = index // self._mul_factor  # % len(self.project_paths)
        image, label = self._load_data(index)
        if self.label_transform is not None:
            label = self.label_transform(label)
        image = ensure_tensor_with_channels(
            image, ndim=self._ndim, dtype=self.dtype
        )
        label = ensure_tensor_with_channels(
            label, ndim=self._ndim, dtype=self.label_dtype
        )
        return image, label

    def _load_data(self, index):
        if index not in self._cache:
            proj = load_data(self.project_paths[index])
            image = proj.image.get_data_by_axis(
                t=0, z=0 if self._ndim == 2 else slice(None)
            )
            label = (
                proj.roi_info.roi
                if self.label_name == "ROI"
                else proj.roi_info.alternative[self.label_name]
            )

            # self._cache[index] = image, label
        else:
            image, label = self._cache[index]
        bb = self._sample_bounding_box(image.shape[-self._ndim :])
        image = image[(slice(None),) + bb]
        label = label[0][bb]
        return image, label

    def __len__(self) -> int:
        return len(self.project_paths) * self._mul_factor

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError(
                "Image padding is not supported yet. Data shape {shape},"
                " patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(
            slice(start, start + psh)
            for start, psh in zip(bb_start, self.patch_shape)
        )


def get_partseg_loader(
    path: str,
    split: str,
    label_name: str,
    patch_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    label_dtype: torch.dtype = torch.float32,
    label_transform=None,
    raw_transform=None,
    mul_factor=16,
    **kwargs
):
    data_path = os.path.join(path, split)
    files = glob(os.path.join(data_path, "*.tgz"))

    if raw_transform is None:
        raw_transform = get_raw_transform()

    ds = PartSegProjectDataset(
        project_paths=files,
        label_name=label_name,
        patch_shape=patch_shape,
        dtype=dtype,
        label_dtype=label_dtype,
        label_transform=label_transform,
        mul_factor=mul_factor,
        raw_transform=raw_transform,
    )
    return get_data_loader(ds, **kwargs)
