import os
import SimpleITK as sitk
from fetpype_utils.utils import n4_bias_field_correction_single
from tqdm import tqdm

def correct_bias_field(input_stacks,  input_masks, output_dir ):
    stack_dir = os.path.join(output_dir)
    os.makedirs(stack_dir, exist_ok=True)
    if input_masks is not None:
        mask_dir = os.path.join(output_dir)
        os.makedirs(mask_dir, exist_ok=True)
    else:
        input_masks = [None] * len(input_stacks)
    
    stacks_out = []
    masks_out = []

    
    for im, mask in tqdm(zip(input_stacks, input_masks)):
        error = False
        try:
            stack_out = os.path.join(
                stack_dir, os.path.basename(im)
            )
            
            sitk_im = sitk.ReadImage(im)

            if mask is not None:
                mask_out = os.path.join(
                    mask_dir,
                    os.path.basename(mask),
                )
                sitk_mask = sitk.ReadImage(mask)

                im_corr = n4_bias_field_correction_single(
                    sitk.Cast(sitk_im, sitk.sitkFloat32),
                    sitk.Cast(sitk_mask, sitk.sitkUInt8),
                )
            else:
                im_corr = n4_bias_field_correction_single(
                    sitk.Cast(sitk_im, sitk.sitkFloat32))

            corrected_sitk_im = sitk.GetImageFromArray(im_corr)
            corrected_sitk_im.CopyInformation(sitk_im)

            sitk.WriteImage(corrected_sitk_im, stack_out)
            if mask is not None:
                sitk.WriteImage(sitk_mask, mask_out)
        except Exception as e:
            print(
                f"Error in bias correction -- Skipping the stack {os.path.basename(im)}: {e}"
            )
            error = True
        if not error:
            stacks_out.append(stack_out)
            if mask is not None:
                masks_out.append(mask_out)
    
    if len(stacks_out) == 0:
        raise ValueError(
            "All stacks (and masks) were discarded during bias field correction."
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run N4 bias field correction")
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
        required=True,
        help="Output directory for corrected stacks and masks",
    )
    args = parser.parse_args()

    correct_bias_field(args.input_stacks, args.input_masks, args.output_dir)


if __name__ == "__main__":
    main()