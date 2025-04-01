import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import subprocess

def denoise(input_stacks, output_dir, n_proc ):
    stack_dir = os.path.join(output_dir)
    os.makedirs(stack_dir, exist_ok=True)
    def process_stack(im):
        try:
            stack_out = os.path.join(
                stack_dir, os.path.basename(im)
            )
            
            subprocess.run(
                ["DenoiseImage", "-i", im, "-n", "Gaussian", "-o", stack_out, "-s", "1"],
                check=True
            )
        except Exception as e:
            print(
                f"Error in denoising -- Skipping the stack {os.path.basename(im)}: {e}"
            )

    with ThreadPoolExecutor(max_workers=n_proc) as executor:
        list(tqdm(executor.map(process_stack, input_stacks), total=len(input_stacks)))

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
        required=True,
        help="Output directory for corrected stacks and masks",
    )

    parser.add_argument(
        "--n_proc",
        type=int,
        default=4,
        help="Number of processes to use for denoising",
    )
    args = parser.parse_args()

    denoise(args.input_stacks, args.output_dir, args.n_proc)

if __name__ == "__main__":
    main()