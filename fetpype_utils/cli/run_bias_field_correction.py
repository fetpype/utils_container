import os
import SimpleITK as sitk
from fetpype_utils.utils import (
    n4_bias_field_correction_single,
    check_same_folder,
)
from tqdm import tqdm


def correct_bias_field(input_stacks, input_masks, output_dir, output_stacks):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(output_stacks[0])

    if input_masks is None:
        input_masks = [None] * len(input_stacks)

    stacks_out = []
    masks_out = []

    for i, (im, mask) in tqdm(enumerate(zip(input_stacks, input_masks))):
        error = False
        try:
            if output_stacks is None:
                stack_out = os.path.join(output_dir, os.path.basename(im))
            else:
                stack_out = output_stacks[i]

            sitk_im = sitk.ReadImage(im)

            if mask is not None:
                sitk_mask = sitk.ReadImage(mask)

                im_corr = n4_bias_field_correction_single(
                    sitk.Cast(sitk_im, sitk.sitkFloat32),
                    sitk.Cast(sitk_mask, sitk.sitkUInt8),
                )
            else:
                im_corr = n4_bias_field_correction_single(
                    sitk.Cast(sitk_im, sitk.sitkFloat32)
                )

            corrected_sitk_im = sitk.GetImageFromArray(im_corr)
            corrected_sitk_im.CopyInformation(sitk_im)

            sitk.WriteImage(corrected_sitk_im, stack_out)
        except Exception as e:
            print(
                f"Error in bias correction -- Skipping the stack {os.path.basename(im)}: {e}"
            )
            error = True
        if not error:
            stacks_out.append(stack_out)

    if len(stacks_out) == 0:
        raise ValueError(
            "All stacks were discarded during bias field correction."
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run N4 bias field correction"
    )
    parser.add_argument(
        "--input_stacks",
        type=str,
        nargs="+",
        required=True,
        help="Input stacks to correct",
    )
    parser.add_argument(
        "--input_masks",
        type=str,
        nargs="+",
        help="Input masks to correct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for corrected stacks and masks",
    )

    parser.add_argument(
        "--output_stacks",
        type=str,
        nargs="+",
        default=None,
        help="Output corrected stacks",
    )
    args = parser.parse_args()
    if args.output_dir is None and args.output_stacks is None:
        raise ValueError(
            "Please provide either an output directory or output stacks."
        )
    if args.output_dir is not None and args.output_stacks is not None:
        raise ValueError(
            "Please provide either an output directory or output stacks, not both."
        )
    if args.output_stacks:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_stacks[0]), exist_ok=True)
        check_same_folder(args.output_stacks)
    correct_bias_field(
        args.input_stacks,
        args.input_masks,
        args.output_dir,
        args.output_stacks,
    )


if __name__ == "__main__":
    main()
