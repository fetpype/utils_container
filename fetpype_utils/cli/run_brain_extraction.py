# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

def print_title(text, center=True, char="-"):
    try:
        terminal_size = os.get_terminal_size().columns
    except Exception:
        terminal_size = 80
    char_length = min(len(text) + 10, terminal_size)
    chars = char * char_length
    text = text.upper()
    if center:
        chars = chars.center(terminal_size)
        text = text.center(terminal_size)
    print("\n" + chars + "\n" + text + "\n" + chars + "\n")

def run_brain_extraction(in_dir, out_dir, method):

    if method == "monaifbs":
        from fetpype_utils.monaifbs.src.inference.monai_dynunet_inference import run_inference
        import fetpype_utils
        from fetpype_utils import monaifbs
        import yaml

        config_file = os.path.join(
            *[
                os.path.dirname(monaifbs.__file__),
                "config",
                "monai_dynUnet_inference_config.yml",
            ]
        )
        if method == "monaifbs":
            brain_ckpt = os.path.join(os.path.dirname(fetpype_utils.__file__),"models_ckpt/monaifbs_checkpoint_dynUnet_DiceXent.pt")
        if not os.path.isfile(brain_ckpt):
            raise ValueError(f"MONAIfbs ckpt not found at {brain_ckpt}.")

        with open(config_file) as f:
            print("*** Config file")
            print(config_file)
            config = yaml.load(f, Loader=yaml.FullLoader)

        # add the output directory to the config dictionary
        config["output"] = {
            "out_postfix": "mask",
            "out_dir": out_dir,
        }
        os.makedirs(config["output"]["out_dir"], exist_ok=True)
        config["inference"]["model_to_load"] = brain_ckpt
        in_files = [os.path.join(in_dir,p) for p in os.listdir(in_dir) if p.endswith(".nii.gz")]
        run_inference(in_files, config)
    else:
        from fetpype_utils.fetal_bet.codes.inference import inference
        import fetpype_utils
        import torch
        # Make it a Namespace
        from types import SimpleNamespace
        args = SimpleNamespace()
        args.n_gpu = 1
        args.deterministic = 1
        args.seed = 1234
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.saved_model_path = os.path.join(os.path.dirname(fetpype_utils.__file__),"models_ckpt/fetbet_AttUNet.pth")
        args.data_path = in_dir
        args.save_path = out_dir

        inference(args)
        



def main():
    import argparse
    import os
    from pathlib import Path

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and computes the brain masks using MONAIfbs "
            " (https://github.com/gift-surg/MONAIfbs/tree/main). Save the masks"
            " into the `masks_dir` folder, follwing the same hierarchy as the `bids_dir`"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input_dir",
        help="Input files.",
        required=True,
    )

    p.add_argument(
        "--out_dir",
        help="Root of the BIDS directory where brain masks will be stored.",
        required=True,
    )

    p.add_argument(
        "--method",
        choices=["monaifbs","fet_bet"],
        default="monaifbs",
    )


    args = p.parse_args()
    print_title("Running Brain extraction")

    input_dir = args.input_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    # Creating the pattern to save the resulting masks

    # Create a tmp directory for the output of monaifbs segmentation
    #masks_tmp = os.path.join(args.masks_dir, "tmp")
    #files_paths = [o[3] for o in files_filtered]
    run_brain_extraction(input_dir, out_dir, args.method)
    # Move files to their definitive location
    #bidsify_monaifbs(files_filtered, bids_layout, mask_pattern, masks_tmp)

    return 0


if __name__ == "__main__":
    main()
