# Copyright 2020 Marta Bianca Maria Ranzini and contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
# \file       custom_transform.py
# \brief      contains a series of custom dict transforms to be used in
#             MONAI data preparation for the dynUnet model
#
# \author     Thomas Sanchez
# \author     Original author - Marta B M Ranzini (marta.ranzini@kcl.ac.uk)
# \date       March 2025

import numpy as np
import copy
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

from monai.config import KeysCollection
from monai.transforms import DivisiblePad, MapTransform, Spacing, Spacingd
from monai.utils import (
    NumpyPadMode,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
import torch

NumpyPadModeSequence = Union[
    Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str
]
GridSampleModeSequence = Union[
    Sequence[Union[GridSampleMode, str]], GridSampleMode, str
]
GridSamplePadModeSequence = Union[
    Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str
]
InterpolateModeSequence = Union[
    Sequence[Union[InterpolateMode, str]], InterpolateMode, str
]


class ConverToOneHotd(MapTransform):
    """
    Convert multi-class label to One Hot Encoding
    """

    def __init__(self, keys, labels):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            labels: list of labels to be converted to one-hot

        """
        super().__init__(keys)
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = list()
            for n in self.labels:
                result.append(d[key] == n)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class MinimumPadd(MapTransform):
    """
    Pad the input data, so that the spatial sizes are at least of size `k`.
    Dictionary-based wrapper of :py:class:`monai.transforms.DivisiblePad`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        k: Union[Sequence[int], int],
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input
                spatial dimensions.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``,
                    ``"maximum"``, ``"mean"``, ``"median"``,
                    ``"minimum"``, ``"reflect"``, ``"symmetric"``,
                    ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function.
                Defaults to ``"constant"``.
                See also:
                https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element
                corresponds to a key in ``keys``.
        See also :py:class:`monai.transforms.SpatialPad`
        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.k = k
        self.padder = DivisiblePad(k=k)

    def __call__(self, data):
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            spatial_shape = np.array(d[key].shape[1:])
            k = np.array(fall_back_tuple(self.k, (1,) * len(spatial_shape)))
            if np.any(spatial_shape < k):
                d[key] = self.padder(d[key], mode=m)
        return d


class InPlaneSpacingd(Spacingd):
    """
    Performs the same operation as the MONAI Spacingd transform but
    allows preserving spacing along some axes, which should be
    indicated as -1.0 in the input `pixdim`.
    E.g. `pixdim=(0.8, 0.8, -1.0)` will change the x-y plane
    spacing while preserving the original z spacing.

    Supports **lazy execution** in MONAI's modern interface.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        pixdim: Union[Sequence[float], float],  # ✅ FIXED
        diagonal: bool = False,
        mode: Union[
            Sequence[Union[str, GridSampleMode]], Union[str, GridSampleMode]
        ] = GridSampleMode.BILINEAR,
        padding_mode: Union[
            Sequence[Union[str, GridSamplePadMode]],
            Union[str, GridSamplePadMode],
        ] = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Union[Sequence[np.dtype], np.dtype] = np.float64,
        scale_extent: bool = False,
        recompute_affine: bool = False,
        min_pixdim: Union[Sequence[float], float, None] = None,
        max_pixdim: Union[Sequence[float], float, None] = None,
        ensure_same_shape: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            pixdim: output voxel spacing. If `-1.0` is provided, that axis'
             spacing is preserved.
            diagonal: whether to resample the input to have a diagonal
             affine matrix.
            mode: Interpolation mode for resampling.
            padding_mode: Padding mode for outside grid values.
            align_corners: Whether to align corners during interpolation.
            dtype: Data type for resampling computation.
            scale_extent: Whether to compute scale based on full voxel extent.
            recompute_affine: If True, recompute affine to avoid
             quantization errors.
            min_pixdim: Minimal spacing allowed before resampling.
            max_pixdim: Maximal spacing allowed before resampling.
            ensure_same_shape: Ensure that outputs have the same shape when
             inputs do.
            allow_missing_keys: If True, missing keys won't raise an error.
            lazy: If `True`, the transform will be stored in metadata instead
             of applied immediately.
        """
        super().__init__(
            keys=keys,
            pixdim=pixdim,
            diagonal=diagonal,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
            scale_extent=scale_extent,
            recompute_affine=recompute_affine,
            min_pixdim=min_pixdim,
            max_pixdim=max_pixdim,
            ensure_same_shape=ensure_same_shape,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
        )
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.dim_to_keep = np.argwhere(self.pixdim == -1.0)

    def __call__(
        self,
        data: Mapping[Hashable, torch.Tensor],
        lazy: Optional[bool] = None,
    ) -> Dict[Hashable, torch.Tensor]:
        """
        Performs spacing transformation while preserving specified dimensions.

        Args:
            data: Dictionary of image tensors and metadata.
            lazy: Overrides lazy execution flag for this call.

        Returns:
            Dictionary with transformed data.
        """
        d = dict(data)

        for key in self.key_iterator(d):

            meta_data = d[key].meta

            current_pixdim = copy.deepcopy(self.pixdim)
            original_pixdim = meta_data["pixdim"]
            old_pixdim = original_pixdim[1:4]
            current_pixdim[self.dim_to_keep] = old_pixdim[self.dim_to_keep]

            spacing_transform = Spacing(
                current_pixdim,
                diagonal=self.diagonal,
                mode=self.mode[self.keys.index(key)],
                padding_mode=self.padding_mode[self.keys.index(key)],
                align_corners=self.align_corners[self.keys.index(key)],
                dtype=self.dtype[self.keys.index(key)],
                scale_extent=self.scale_extent[self.keys.index(key)],
                recompute_affine=False,
            )

            d[key] = spacing_transform(data_array=d[key])

        return d


class RestoreOriginalSpacingd(MapTransform):
    def __init__(
        self, keys: Sequence[Hashable], allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            meta_data = d[key].meta

            original_pixdim = meta_data["pixdim"].squeeze()

            # Resample back to original spacing
            spacing_transform = Spacing(
                pixdim=original_pixdim[1:4],  # Exclude batch dimension
                diagonal=False,
                mode="nearest",  # Use nearest for mask
                align_corners=False,
                recompute_affine=False,
            )
            d[key] = spacing_transform(data_array=d[key])

        return d
