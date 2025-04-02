import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import subprocess


def denoise(input_stacks, output_dir, output_stacks, n_proc):

    if output_dir:
        stack_dir = os.path.join(output_dir)
        os.makedirs(stack_dir, exist_ok=True)

    def process_stack(im, stack_out):
        try:
            if stack_out is None:
                stack_out = os.path.join(stack_dir, os.path.basename(im))

            subprocess.run(
                [
                    "DenoiseImage",
                    "-i",
                    im,
                    "-n",
                    "Gaussian",
                    "-o",
                    stack_out,
                    "-s",
                    "1",
                ],
                check=True,
            )
        except Exception as e:
            print(
                f"Error in denoising -- Skipping the stack {os.path.basename(im)}: {e}"
            )

    with ThreadPoolExecutor(max_workers=n_proc) as executor:
        list(
            tqdm(
                executor.map(process_stack, input_stacks, output_stacks),
                total=len(input_stacks),
            )
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run ANTS denoising")
    parser.add_argument(
        "--input_stacks",
        type=str,
        nargs="+",
        required=True,
        help="Input stacks to correct",
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

    parser.add_argument(
        "--n_proc",
        type=int,
        default=4,
        help="Number of processes to use for denoising",
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
    denoise(
        args.input_stacks, args.output_dir, args.output_stacks, args.n_proc
    )


if __name__ == "__main__":
    main()
