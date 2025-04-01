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
# \file       monai_dynunet_inference.py
# \brief      Script to perform automated fetal brain segmentation using a
#             pre-trained dynUNet model in MONAI
#               Example config file required by the main function is shown in
#               monaifbs/config/monai_dynUnet_inference_config.yml
#               Example of model loaded by this evaluation function is
#               stored in monaifbs/models/checkpoint_dynUnet_DiceXent.pt
#
# \author     Thomas Sanchez (thomas.sanchez@unil.ch)
# \author     Marta B M Ranzini (marta.ranzini@kcl.ac.uk)
# \date       March 2025
#

import os
import logging
import sys
import yaml
import torch
from monai.config import print_config
from fetpype_utils.monaifbs.models.old_dynunet import DynUNet
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Activationsd,
    AsDiscreted,
    SaveImaged,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    Spacingd,
)
from monai.inferers import SlidingWindowInferer
import numpy as np
from tqdm import tqdm


def affine_norm(affine):
    return np.sqrt((affine[0:3, 0:3] ** 2).sum(0))


def run_inference(input_data, config_info):
    print(f"Running on {input_data}")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print("*** MONAI config: ")
    print_config()

    print("*** Network inference config: ")
    print(yaml.dump(config_info))

    nr_out_channels = config_info["inference"]["nr_out_channels"]
    spacing = config_info["inference"]["spacing"]
    prob_thr = config_info["inference"]["probability_threshold"]
    model_to_load = config_info["inference"]["model_to_load"]
    patch_size = config_info["inference"]["inplane_size"] + [1]

    if not os.path.exists(model_to_load):
        raise FileNotFoundError("Trained model not found")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(
                keys=["image"], nonzero=False, channel_wise=True
            ),
            Spacingd(
                keys=["image"],
                pixdim=(0.8, 0.8, -1.0),
                mode="bilinear",
                padding_mode="zeros",
            ),
        ]
    )

    print("***  Preparing network ... ")
    # automatically extracts the strides and kernels
    # based on nnU-Net empirical rules
    spacings = spacing[:2]
    sizes = patch_size[:2]
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=nr_out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name=("instance", {"affine": True}),
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
    )
    net.load_state_dict(torch.load(model_to_load, map_location="cpu")["net"])
    net.to(device)
    net.eval()

    post_transforms = Compose(
        [
            Activationsd(
                keys="image",
                sigmoid=nr_out_channels == 1,
                softmax=nr_out_channels > 1,
            ),
            AsDiscreted(
                keys="image",
                argmax=True,
                threshold_values=True,
                logit_thresh=prob_thr,
            ),
            KeepLargestConnectedComponentd(keys="image", applied_labels=1),
        ]
    )

    inferer = SlidingWindowInferer(
        roi_size=[448, 512], sw_batch_size=4, overlap=0.0
    )
    print("*** Running inference...")

    loader = LoadImaged(
        keys=["image"], reader="NibabelReader", image_only=False
    )

    with torch.no_grad():
        for data in tqdm(input_data):

            im = loader({"image": data})
            meta = im["image"].meta
            im = val_transforms(im)["image"].to(device)

            batch_2d = im.permute(3, 0, 1, 2).contiguous()

            # Perform inference with TTA (flipping)
            pred = inferer(batch_2d, net)  # Forward pass on all slices
            flip_pred_1 = torch.flip(
                inferer(torch.flip(batch_2d, dims=(2,)), net), dims=(2,)
            )
            flip_pred_2 = torch.flip(
                inferer(torch.flip(batch_2d, dims=(3,)), net), dims=(3,)
            )
            flip_pred_3 = torch.flip(
                inferer(torch.flip(batch_2d, dims=(2, 3)), net), dims=(2, 3)
            )

            # Average predictions
            pred = (pred + flip_pred_1 + flip_pred_2 + flip_pred_3) / 4

            pred_3d = pred.permute(1, 2, 3, 0)
            post = post_transforms({"image": pred_3d})

            pixdim = affine_norm(meta["original_affine"]).tolist() + [-1.0]
            invert = Spacingd(
                keys=["image"],
                pixdim=pixdim + [-1.0],
                mode="nearest",
            )

            post = invert(post)
            post["image"].meta = meta
            saver = SaveImaged(
                keys="image",
                output_dir=config_info["output"]["out_dir"],
                print_log=False,
                separate_folder=False,
                output_postfix="",
                output_ext=".nii.gz",
                resample=False,
            )
            saver(post)

    print("Done!")
