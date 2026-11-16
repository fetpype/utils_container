"""
Single entry point for fetal-BET: inspects each input file's header to pick a pipeline
(2D for a thick-slice stack, 3D for a near-isotropic volume, 4D for a time/direction series)
and dispatches to inference_2d.py/inference_3d.py/inference_4d.py. --pipeline can force 2d/3d
for 3D data; 4D data can only use 4d. --input_path may be a single file or a directory (mixed
dimensionalities are grouped by pipeline and batched into one call per pipeline).

Usage
-----
    python inference.py --input_path sub-01_t2w.nii.gz --save_path out/
    python inference.py --input_path data/ --save_path out/
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from glob import glob

import nibabel as nib
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

ISOTROPIC_TOL = 1.05  # matches mask_refine.py's own threshold for "no clear thick axis"

PIPELINE_SCRIPTS = {
    "2d": "inference_2d.py",
    "3d": "inference_3d.py",
    "4d": "inference_4d.py",
}


def _effective_shape(shape):
    """Drop trailing singleton dims beyond the 3rd, so a plain 3D volume with a redundant
    size-1 4th axis isn't misrouted to 4D."""
    shape = tuple(shape)
    while len(shape) > 3 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def detect_pipeline(path, isotropic_tol=ISOTROPIC_TOL):
    """Inspect a NIfTI header (no voxel data read) and return (ndim, auto_pipeline, reason)."""
    img = nib.load(path)
    shape = _effective_shape(img.shape)
    ndim = len(shape)

    if ndim == 4:
        return ndim, "4d", f"4D input (shape={shape}) - a time/direction series"

    if ndim != 3:
        raise ValueError(f"Unsupported data dimensionality {ndim} for {path} (expected 3D or 4D)")

    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    order = np.argsort(-zooms)
    ratio = zooms[order[0]] / zooms[order[1]]
    if ratio > isotropic_tol:
        return ndim, "2d", (
            f"3D input with a clear thick axis (zooms={tuple(zooms)}, ratio={ratio:.2f} > "
            f"{isotropic_tol}) - treated as a thick-slice stack"
        )
    return ndim, "3d", (
        f"3D input with near-isotropic spacing (zooms={tuple(zooms)}, ratio={ratio:.2f} <= "
        f"{isotropic_tol}) - treated as a genuine 3D volume"
    )


def resolve_pipeline(path, pipeline_arg):
    """Combine header detection with --pipeline, validating a forced choice is compatible
    with the data. Returns (pipeline, reason)."""
    ndim, auto_pipeline, reason = detect_pipeline(path)

    if pipeline_arg == "auto":
        return auto_pipeline, reason

    if ndim == 4 and pipeline_arg != "4d":
        raise ValueError(
            f"{path} is 4D; only --pipeline 4d can process it "
            f"(--pipeline {pipeline_arg} has no way to handle a time/direction axis)."
        )
    if ndim == 3 and pipeline_arg == "4d":
        raise ValueError(
            f"{path} is 3D; --pipeline 4d expects a 4D series. "
            f"Use --pipeline 2d or 3d (or auto) instead."
        )
    return pipeline_arg, f"forced via --pipeline {pipeline_arg} (header says: {reason})"


def build_command(pipeline, input_dir, save_path, args):
    script_path = os.path.join(SRC_DIR, PIPELINE_SCRIPTS[pipeline])
    # input_dir is a throwaway symlink dir; display_path shows the user's real --input_path instead.
    cmd = [PYTHON, "-W", "ignore::FutureWarning", script_path,
           "--data_path", input_dir, "--save_path", save_path,
           "--display_path", args.input_path]

    if args.device is not None:
        cmd += ["--device", args.device]
    if args.n_gpu is not None:
        cmd += ["--n_gpu", str(args.n_gpu)]
    if args.refine is not None:
        cmd += ["--refine", str(args.refine)]
    if args.amp is not None:
        if pipeline in ("2d", "3d"):
            cmd += ["--amp", str(args.amp)]
        else:
            print(f"[inference.py] --amp {args.amp} was given but ignored: "
                  f"inference_4d.py has no --amp (4D doesn't go through mixed precision).",
                  flush=True)

    return cmd


def run_pipeline_group(pipeline, paths, save_path, args):
    """Symlink `paths` into one temp directory and make a single call to that pipeline's script."""
    os.makedirs(save_path, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fetalbet_input_") as tmp_dir:
        for path in paths:
            link_path = os.path.join(tmp_dir, os.path.basename(path))
            os.symlink(os.path.abspath(path), link_path)

        cmd = build_command(pipeline, tmp_dir, save_path, args)
        print(f"[inference.py] running {PIPELINE_SCRIPTS[pipeline]} ...", flush=True)
        start = time.time()
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"[inference.py] {PIPELINE_SCRIPTS[pipeline]} finished in {elapsed:.1f}s", flush=True)
        return elapsed


def _mask_path_for(input_path, save_path):
    """The dispatched scripts all save as "<input stem>_mask.nii.gz" (SaveImaged's
    output_postfix="mask" for 2D/3D, the same convention hardcoded in inference_4d.py)."""
    stem = os.path.basename(input_path)[: -len(".nii.gz")]
    return os.path.join(save_path, f"{stem}_mask.nii.gz")


def _make_world_accessible(input_paths, save_path):
    """Containers commonly run as root, so output written to a bind-mounted host directory
    ends up root-owned; chmod just the file(s) this run produced open so the invoking host
    user can read/edit/delete them without needing `docker run --user` or sudo - leaving
    everything else under save_path (e.g. pre-existing files, if save_path==input_path)
    untouched."""
    for input_path in input_paths:
        mask_path = _mask_path_for(input_path, save_path)
        if os.path.exists(mask_path):
            os.chmod(mask_path, 0o666)


def inference(args):
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"--input_path not found: {args.input_path}")

    if os.path.isdir(args.input_path):
        all_paths = sorted(glob(os.path.join(args.input_path, "*.nii.gz")))
        # Skip this tool's own output naming convention so re-running on a save_path==input_path
        # directory doesn't reprocess a previous run's masks as new input.
        paths = [p for p in all_paths if not os.path.basename(p).endswith("_mask.nii.gz")]
        skipped = len(all_paths) - len(paths)
        if skipped:
            print(f"[inference.py] skipping {skipped} file(s) matching *_mask.nii.gz "
                  f"(treated as previous output, not input)", flush=True)
        if not paths:
            raise FileNotFoundError(f"No .nii.gz input files found under: {args.input_path}")

        groups = {}
        for path in paths:
            pipeline, _ = resolve_pipeline(path, args.pipeline)
            groups.setdefault(pipeline, []).append(path)

        total_elapsed = 0.0
        for pipeline, group_paths in groups.items():
            print(f"[inference.py] {len(group_paths)} file(s) -> pipeline = {pipeline}", flush=True)
            total_elapsed += run_pipeline_group(pipeline, group_paths, args.save_path, args)

        print(f"[inference.py] total elapsed: {total_elapsed:.1f}s", flush=True)
        _make_world_accessible(paths, args.save_path)
        return

    pipeline, reason = resolve_pipeline(args.input_path, args.pipeline)
    print(f"[inference.py] {reason}", flush=True)
    print(f"[inference.py] pipeline = {pipeline}", flush=True)

    run_pipeline_group(pipeline, [args.input_path], args.save_path, args)
    _make_world_accessible([args.input_path], args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input_path", type=str, required=True,
                         help="path to a single input .nii.gz file, or a directory of them "
                              "(each file is routed individually; files landing on the same "
                              "pipeline are processed together in one call)")
    parser.add_argument("--save_path", type=str, required=True,
                         help="directory to save the predicted mask")

    parser.add_argument("--pipeline", type=str, default="auto", choices=["auto", "2d", "3d", "4d"],
                         help="which script to dispatch to. 'auto' (default) decides from the "
                              "input's header: 4D -> 4d (the only option that can handle it); "
                              "3D with a clear thick axis -> 2d; 3D near-isotropic -> 3d. 3D "
                              "data may also be forced through either 2d or 3d explicitly; 4D "
                              "data can only use 4d.")

    parser.add_argument("--device", type=str, default=None,
                         help="forwarded to the dispatched script; leave unset to use its own default")
    parser.add_argument("--n_gpu", type=int, default=None,
                         help="forwarded to the dispatched script; leave unset to use its own default")
    parser.add_argument("--refine", type=int, default=None,
                         help="forwarded to the dispatched script; leave unset to use its own default")
    parser.add_argument("--amp", type=int, default=None,
                         help="forwarded to inference_2d.py/inference_3d.py only (inference_4d.py "
                              "has no --amp); leave unset to use the target script's own default")

    args = parser.parse_args()
    inference(args)
