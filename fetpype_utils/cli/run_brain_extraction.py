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
from fetpype_utils.utils import check_same_folder


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


def run_brain_extraction(in_files, out_dir, out_masks, method, device):
    print("RUNNING:", in_files, out_dir, out_masks, method, device)
    if in_files is not None:
        # Check that the files exist
        for f in in_files:
            if not os.path.isfile(f):
                raise ValueError(f"Input file {f} does not exist.")

    if method == "monaifbs":
        from fetpype_utils.monaifbs.src.inference.monai_dynunet_inference import (
            run_inference,
        )
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
            brain_ckpt = os.path.join(
                os.path.dirname(fetpype_utils.__file__),
                "models_ckpt/monaifbs_checkpoint_dynUnet_DiceXent.pt",
            )
        if not os.path.isfile(brain_ckpt):
            raise ValueError(f"MONAIfbs ckpt not found at {brain_ckpt}.")

        with open(config_file) as f:
            print("*** Config file")
            print(config_file)
            config = yaml.load(f, Loader=yaml.FullLoader)

        # add the output directory to the config dictionary
        config["output"] = {
            "out_postfix": "",
            "out_dir": out_dir,
        }
        os.makedirs(config["output"]["out_dir"], exist_ok=True)
        config["inference"]["model_to_load"] = brain_ckpt

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
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        args.saved_model_path = os.path.join(
            os.path.dirname(fetpype_utils.__file__),
            "models_ckpt/fetbet_AttUNet.pth",
        )
        args.data_path = None
        args.data_files = in_files
        args.save_path = out_dir

        inference(args)

    # Then take the out_dir / in_files and rename them:
    if out_masks is None:
        for f in in_files:
            out_name = os.path.join(out_dir, os.path.basename(f))
            out_rename = out_name.replace("_T2w", "_mask")
            # move the file
            os.rename(out_name, out_rename)
    else:
        for f, m in zip(in_files, out_masks):
            out_name = os.path.join(out_dir, os.path.basename(f))
            os.rename(out_name, m)


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
        help="Input directory.",
        default=None,
    )

    p.add_argument(
        "--input_stacks",
        help="Input files.",
        nargs="+",
        default=None,
    )

    p.add_argument(
        "--output_dir",
        help="Root of the BIDS directory where brain masks will be stored.",
        default=None,
    )
    p.add_argument(
        "--output_masks",
        help="Output masks.",
        nargs="+",
        default=None,
    )

    p.add_argument(
        "--method",
        choices=["monaifbs", "fet_bet"],
        default="monaifbs",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    args = p.parse_args()
    print_title(f"Running Brain extraction ({args.method} -- {args.device})")
    if args.input_dir is None and args.input_stacks is None:
        raise ValueError(
            "Please provide either an input directory or input stacks."
        )
    if args.input_dir is not None and args.input_stacks is not None:
        raise ValueError(
            "Please provide either an input directory or input stacks, not both."
        )

    # Allow only selected combinations: input_dir -> output_dir
    # input_stacks -> output_masks

    if args.input_dir:
        assert (
            args.output_dir is not None
        ), "Please provide an output directory if you provide an input directory."

    if args.input_stacks:
        assert (
            args.output_masks is not None
        ), "Please provide output masks if you provide input stacks."

        assert len(args.input_stacks) == len(
            args.output_masks
        ), "Input stacks and output masks should have the same length."

    if args.input_stacks:
        check_same_folder(args.input_stacks)
    if args.output_masks:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_masks[0]), exist_ok=True)

    if args.output_dir is None and args.output_masks is None:
        raise ValueError(
            "Please provide either an output directory or output stacks."
        )
    if args.output_dir is not None and args.output_masks is not None:
        raise ValueError(
            "Please provide either an output directory or output stacks, not both."
        )
    if args.output_masks:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_masks[0])
        check_same_folder(args.output_masks)

        os.makedirs(output_dir, exist_ok=True)

    if args.input_stacks is None:
        input_stacks = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        ]
    else:
        input_stacks = args.input_stacks
    output_dir = args.output_dir if args.output_dir else output_dir
    output_masks = args.output_masks

    os.makedirs(output_dir, exist_ok=True)

    run_brain_extraction(
        input_stacks, output_dir, output_masks, args.method, args.device
    )

    return 0


if __name__ == "__main__":
    main()
