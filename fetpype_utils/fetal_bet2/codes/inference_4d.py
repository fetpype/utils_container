"""
4D (time-series) inference for fetal-BET segmentation.

Uses the same hardcoded 3D AttUNet architecture and sliding-window inferer as the 3D
pipeline (same roi_size/overlap). The only difference is the input - instead of a single
3D volume (H, W, D), each file here is a 4D series (H, W, D, T), e.g. an fMRI run. Each of
the T volumes along the last axis is run through the exact same 3D preprocessing +
sliding-window inference + inversion pipeline, independently, and the T predicted masks
are then stacked back along the time axis into a single 4D (H, W, D, T) mask volume.
"""

from glob import glob
from pathlib import Path

import argparse
import os

import numpy as np
import nibabel as nib
import torch

import monai.transforms as tr
from monai.data import DataLoader, MetaTensor, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import AttentionUnet
from tqdm import tqdm

from mask_refine import detect_slice_axis, refine_volume

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models_ckpt" / "fetbet2_AttUNet4D.pth"
ROI_SIZE = (224, 224, 128)


def display_path_for(basename, display_path, data_path):
    """Map a file's basename back to a user-facing path for printing, using --display_path
    (the caller's real path) instead of this script's own --data_path when set."""
    base = display_path or data_path
    return os.path.join(base, basename) if os.path.isdir(base) else base


def load_checkpoint(model, checkpoint_path, device):
    """Load a checkpoint regardless of whether it (or the current model) is
    DataParallel-wrapped, by matching the "module." prefix on both sides."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    is_wrapped = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = next(iter(state_dict)).startswith("module.")

    if is_wrapped and not has_module_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not is_wrapped and has_module_prefix:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def build_transforms():
    """Same per-volume preprocessing as the 3D pipeline, minus LoadImaged/EnsureChannelFirstd:
    each timepoint is already an in-memory MetaTensor, not a file to load."""
    return [
        tr.Spacingd(keys=["image"], pixdim=(1.0, 1.0, -1), mode="bilinear", padding_mode="zeros"),
        tr.Spacingd(keys=["image"], pixdim=(-1, -1, 1.0), mode="nearest", padding_mode="zeros"),
        tr.ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1, clip=True),
    ]


class TimepointDataset:
    """Lazily slices one timepoint out of a 4D (H, W, D, T) array and preprocesses it. With
    num_workers > 0, the next frame's preprocessing overlaps the current frame's GPU inference."""

    def __init__(self, data, affine, num_timepoints, transform):
        self.data = data
        self.affine = affine
        self.num_timepoints = num_timepoints
        self.transform = transform

    def __len__(self):
        return self.num_timepoints

    def __getitem__(self, idx):
        frame = np.ascontiguousarray(self.data[..., idx])
        volume = MetaTensor(torch.from_numpy(frame)[None], affine=self.affine)
        return self.transform({"image": volume})


def pick_sw_batch_size(model, device, roi_size=ROI_SIZE, target_frac=1.0, max_sw_batch_size=16):
    """Auto-size sw_batch_size from currently-free VRAM. Probes at batch=1 and batch=2 to
    isolate the fixed cost (model weights) from the true per-window marginal cost, then picks
    the largest batch that fits in what's currently free on this (possibly shared) GPU.
    infer_with_oom_fallback() is the safety net if that estimate is still too optimistic."""
    if device == "cpu" or not torch.cuda.is_available():
        return 1

    def peak_mem_for_batch(batch):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        dummy = torch.zeros((batch, 1, *roi_size), device=device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        del dummy
        torch.cuda.empty_cache()
        return peak

    mem_at_1 = peak_mem_for_batch(1)
    mem_at_2 = peak_mem_for_batch(2)
    marginal_bytes = max(mem_at_2 - mem_at_1, 1)
    fixed_bytes = max(mem_at_1 - marginal_bytes, 0)

    free_bytes, _total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
    budget_bytes = free_bytes * target_frac
    sw_batch_size = max(1, min(max_sw_batch_size, int((budget_bytes - fixed_bytes) // marginal_bytes)))

    print(
        f"[auto sw_batch_size] fixed overhead ~{fixed_bytes / 1e9:.2f} GB, "
        f"~{marginal_bytes / 1e9:.2f} GB per additional window, "
        f"{free_bytes / 1e9:.2f} GB currently free -> sw_batch_size={sw_batch_size}"
    )
    return sw_batch_size


def infer_with_oom_fallback(inferer, inputs, model):
    """On CUDA OOM, halve sw_batch_size (down to 1) and retry instead of crashing; the
    reduced value sticks for the rest of the run."""
    while True:
        try:
            return inferer(inputs, model)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if inferer.sw_batch_size <= 1:
                raise
            inferer.sw_batch_size = max(1, inferer.sw_batch_size // 2)
            print(f"[WARN] CUDA out of memory - retrying with sw_batch_size={inferer.sw_batch_size}")


def run_one_file(nii_path, model, inferer, device, save_path, max_timepoints=None, num_workers=2,
                  refine=True):
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D volume (x, y, z, t), got shape {data.shape} for {nii_path}")

    num_timepoints = data.shape[-1] if max_timepoints is None else min(max_timepoints, data.shape[-1])
    affine = torch.as_tensor(img.affine, dtype=torch.float64)

    if refine:
        zooms = np.linalg.norm(img.affine[:3, :3], axis=0)
        slice_axis = detect_slice_axis(zooms, data.shape[:3])

    pre_transforms = tr.Compose(build_transforms())
    post_transforms = tr.Compose([
        tr.Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        tr.Activationsd(keys="pred", softmax=True),
        tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
    ])

    frame_dataset = TimepointDataset(data, affine, num_timepoints, pre_transforms)
    loader = DataLoader(
        frame_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    mask_frames = []
    with torch.no_grad(), tqdm(total=num_timepoints, desc=os.path.basename(nii_path)) as pbar:
        for batch in loader:
            sample = decollate_batch(batch)[0]

            inputs = batch["image"].to(device, non_blocking=True)
            sample["pred"] = infer_with_oom_fallback(inferer, inputs, model)[0]

            sample = post_transforms(sample)
            frame_mask = sample["pred"][0].cpu().numpy() > 0
            if refine:
                # frames are refined independently, never chained across time
                frame_mask = refine_volume(frame_mask, slice_axis)["refined"]
            mask_frames.append(frame_mask.astype(np.uint8))

            pbar.update(1)

    mask_4d = np.stack(mask_frames, axis=-1)  # (H, W, D, T)

    os.makedirs(save_path, exist_ok=True)
    out_name = os.path.basename(nii_path).replace(".nii.gz", "") + "_mask.nii.gz"
    out_path = os.path.join(save_path, out_name)
    nib.save(nib.Nifti1Image(mask_4d, img.affine, img.header), out_path)
    return out_path


def inference(args):
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        dropout=0,
    )

    device = args.device
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    load_checkpoint(model, args.saved_model_path, device)
    model.eval()

    # worth it here since the model runs many times per file at the same input shape
    torch.backends.cudnn.benchmark = True

    sw_batch_size = args.sw_batch_size if args.sw_batch_size is not None else pick_sw_batch_size(model, device)
    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE, sw_batch_size=sw_batch_size, overlap=0.25, mode="gaussian"
    )

    nii_files = sorted(glob(os.path.join(args.data_path, "*.nii.gz")))
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found under: {args.data_path}")

    print("\n" + "=" * 60)
    print("Inference summary")
    print("-" * 60)
    print(f"Input data path : {args.display_path or args.data_path}")
    print(f"Output save path: {args.save_path}")
    print(f"Number of cases : {len(nii_files)} (.nii.gz 4D series)")
    if args.max_timepoints is not None:
        print(f"Timepoints/case : {args.max_timepoints} (truncated via --max_timepoints)")
    print("=" * 60 + "\n")

    for i, nii_path in enumerate(nii_files):
        shape = nib.load(nii_path).shape
        display_path = display_path_for(os.path.basename(nii_path), args.display_path, args.data_path)
        print(f"[{i + 1}/{len(nii_files)}] processing: {display_path} (shape={shape})")
        out_path = run_one_file(
            nii_path, model, inferer, device, args.save_path,
            args.max_timepoints, args.num_workers, bool(args.refine),
        )
        print(f"Saved 4D mask: {out_path}")

    print('Process completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4D (time-series) inference for fetal-BET segmentation"
    )

    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )

    parser.add_argument(
        "--saved_model_path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the saved 3D model checkpoint"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input 4D NIfTI files (.nii.gz, shape H x W x D x T)"
    )

    parser.add_argument(
        "--display_path",
        type=str,
        default=None,
        help='shown as "Input data path" in the summary instead of --data_path (e.g. '
             'inference.py passes the original path it was given, since --data_path here is '
             'really just a throwaway symlink directory it built); leave unset to show '
             '--data_path as-is'
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save output 4D masks (.nii.gz)"
    )

    parser.add_argument(
        "--max_timepoints",
        type=int,
        default=None,
        help="If set, only process the first N timepoints per file (useful for a quick smoke test)"
    )

    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=None,
        help="Sliding-window batch size: how many overlapping 3D windows are run through the "
             "model at once per timepoint. Faster at higher values but much more VRAM-hungry. "
             "Default (unset) auto-detects this GPU's currently-free VRAM and picks a value "
             "that uses all of it; pass an explicit value to override, e.g. for a known-small "
             "or heavily shared GPU."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Background worker processes that preprocess upcoming timepoints while the GPU is "
             "busy with the current one. Defaults to 0 (avoids Docker's default /dev/shm limit "
             "crashing worker IPC); raise it if your environment allows more shared memory."
    )

    parser.add_argument(
        "--refine",
        type=int,
        default=1,
        help="apply mask_refine.py's connected-component cleanup to each frame's predicted "
             "mask before stacking/saving (removes small disconnected mis-segmentation while "
             "keeping ambiguous competing candidates; frames are refined independently, never "
             "chained across time). Set to 0 to save the raw predicted mask instead. Either "
             "way, exactly one 4D mask file is written."
    )

    args = parser.parse_args()
    inference(args)
