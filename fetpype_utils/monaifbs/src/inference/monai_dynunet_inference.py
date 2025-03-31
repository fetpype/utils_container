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
# \brief      Script to perform automated fetal brain segmentation using a pre-trained dynUNet model in MONAI
#               Example config file required by the main function is shown in
#               monaifbs/config/monai_dynUnet_inference_config.yml
#               Example of model loaded by this evaluation function is stored in
#               monaifbs/models/checkpoint_dynUnet_DiceXent.pt
#
# \author     Marta B M Ranzini (marta.ranzini@kcl.ac.uk)
# \date       November 2020
#

import os
import sys
import yaml
import argparse
import logging
import torch

from torch.utils.data import DataLoader

from monai.config import print_config
from monai.data import DataLoader, Dataset
from fetpype_utils.monaifbs.models.old_dynunet import DynUNet
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, StatsHandler, PostProcessing
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ToTensord,
    SqueezeDimd,
    ToMetaTensord,
    Activationsd,
    AsDiscreted,
    SaveImaged,
    Spacingd,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
)
from monai.transforms import SaveImage
from fetpype_utils import monaifbs
from fetpype_utils.monaifbs.src.utils.custom_inferer import SlidingWindowInferer2D
from fetpype_utils.monaifbs.src.utils.custom_transform import InPlaneSpacingd, RestoreOriginalSpacingd

import os
import torch
import yaml
import logging
import sys
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Spacingd, ToMetaTensord, Activationsd, AsDiscreted, KeepLargestConnectedComponentd, SaveImaged
)
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer

def create_data_list_of_dictionaries(input_files):
    """
    Convert the list of input files to be processed in the dictionary format needed for MONAI
    Args:
        input_files: str or list of strings, filenames of images to be processed
    Returns:
        full_list: list of dicts, storing the filenames input to the inference pipeline
    """

    print("*** Input data: ")
    full_list = []
    # convert to list if single file
    if type(input_files) is str:
        input_files = [input_files]
    for current_f in input_files:
        if os.path.isfile(current_f):
            print(current_f)
            full_list.append({"image": current_f})
        else:
            raise FileNotFoundError(
                "Expected image file: {} not found".format(current_f)
            )
    return full_list

def run_inference(input_data, config_info):
    print(f"Running on {input_data}")
    val_files = create_data_list_of_dictionaries(input_data)
    
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
    
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        InPlaneSpacingd(
                keys=["image"],
                pixdim=spacing,
                mode="bilinear",
            ),
        ToMetaTensord(keys=["image"]),
    ])
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=config_info["device"]["num_workers"])
    
    print("***  Preparing network ... ")
    # automatically extracts the strides and kernels based on nnU-Net empirical rules
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
    
    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=nr_out_channels == 1, softmax=nr_out_channels > 1),
        AsDiscreted(keys="pred", argmax=True, threshold_values=True, logit_thresh=prob_thr),
        KeepLargestConnectedComponentd(keys="pred", applied_labels=1),
        RestoreOriginalSpacingd(
            keys="pred",
        ),
        SaveImaged(keys="pred", output_dir="./output", output_postfix="test"),
    ])
    
    # Reshape [1, 1, H, W, D] → [D, 1, H, W] (batch of 2D slices)


    inferer = SlidingWindowInferer(roi_size=[448,512], sw_batch_size=4, overlap=0.0)
    # import pdb
    # print("*** Running inference...")
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            depth = image.shape[-1]
            batch_2d = image.permute(4, 0, 1, 2, 3).contiguous()  # (D, 1, H, W)

            # Sliding window inferer (now operates on all slices at once)
            
            batch_2d = batch_2d.squeeze(1)
            # Perform inference with TTA (flipping)
            print("BATCH SIZE", batch_2d.shape)
            
            pred = inferer(batch_2d, net)  # Forward pass on all slices
            flip_pred_1 = torch.flip(inferer(torch.flip(batch_2d, dims=(2,)), net), dims=(2,))
            flip_pred_2 = torch.flip(inferer(torch.flip(batch_2d, dims=(3,)), net), dims=(3,))
            flip_pred_3 = torch.flip(inferer(torch.flip(batch_2d, dims=(2, 3)), net), dims=(2, 3))

            # Average predictions
            pred = (pred + flip_pred_1 + flip_pred_2 + flip_pred_3) / 4

            # Reshape back to [1, 1, H, W, D]
            pred_3d = pred.permute(1, 2, 3, 0)  # (1, 1, H, W, D)
            pred_3d.meta["affine"] = pred_3d.meta["affine"].squeeze(0)  # Convert from [1, 4, 4] to [4, 4]


            batch = post_transforms({"pred": pred_3d})
    
    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run inference with dynUnet with MONAI."
    )
    parser.add_argument(
        "--in_files",
        dest="in_files",
        metavar="in_files",
        type=str,
        nargs="+",
        help="all files to be processed",
        required=True,
    )
    parser.add_argument(
        "--out_folder",
        dest="out_folder",
        metavar="out_folder",
        type=str,
        help="directory where to store the outputs",
        required=True,
    )
    parser.add_argument(
        "--out_postfix",
        dest="out_postfix",
        metavar="out_postfix",
        type=str,
        help="postfix to add to the input names for the output filename",
        default="seg",
    )
    parser.add_argument(
        "--config_file",
        dest="config_file",
        metavar="config_file",
        type=str,
        help="config file containing network information for inference",
        default=None,
    )
    args = parser.parse_args()

    # check existence of config file and read it
    config_file = args.config_file
    if config_file is None:
        config_file = os.path.join(
            *[
                os.path.dirname(monaifbs.__file__),
                "config",
                "monai_dynUnet_inference_config.yml",
            ]
        )
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            "Expected config file: {} not found".format(config_file)
        )
    with open(config_file) as f:
        print("*** Config file")
        print(config_file)
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read the input files
    in_files = args.in_files

    # add the output directory to the config dictionary
    config["output"] = {
        "out_postfix": args.out_postfix,
        "out_dir": args.out_folder,
    }
    if not os.path.exists(config["output"]["out_dir"]):
        os.makedirs(config["output"]["out_dir"])

    if config["inference"]["model_to_load"] == "default":
        config["inference"]["model_to_load"] = os.path.join(
            *[
                os.path.dirname(monaifbs.__file__),
                "models",
                "checkpoint_dynUnet_DiceXent.pt",
            ]
        )
    print(in_files, config)
    # run inference with MONAI dynUnet
    run_inference(in_files, config)
