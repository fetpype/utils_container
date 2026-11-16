"""
Inference for brain-extraction models trained on the SVR-reconstructed volumes. Given a
directory of .nii.gz volumes, runs sliding-window inference with the AttUNet3D architecture
and saves predicted brain masks.

Model architecture, voxel spacing, and sliding-window settings are hardcoded here to match
the checkpoint this script loads by default (models/AttUNet3D.pth).
"""

from pathlib import Path
import argparse
import os
from glob import glob

import numpy as np
import nibabel as nib
import torch

import monai.transforms as tr
from monai.data import DataLoader, MetaTensor, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import AttentionUnet
from monai.transforms import SaveImaged, MapTransform
from tqdm import tqdm

from mask_refine import refine_volume_isotropic

PROJECT_DIR = Path(__file__).resolve().parent.parent

VOXEL_SPACING = (0.8, 0.8, 0.8)
IMG_SIZE = (128, 128, 128)
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.25


class RefineMaskd(MapTransform):
    """Cleans up small disconnected mis-segmented components using mask_refine.py's isotropic
    (erosion-based 3D) algorithm - this data has no privileged slice axis to chain along."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            pred = d[key]
            arr = pred.as_tensor() if hasattr(pred, "as_tensor") else pred
            mask = arr[0].cpu().numpy() > 0  # drop channel dim -> (H, W, D)
            refined = refine_volume_isotropic(mask)["refined"]
            new_data = torch.from_numpy(refined).to(arr.dtype)[None]
            d[key] = MetaTensor(new_data, meta=pred.meta) if hasattr(pred, "meta") else new_data
        return d


def display_path_for(basename, display_path, data_path):
    """Map a file's basename back to a user-facing path for printing, using --display_path
    (the caller's real path) instead of this script's own --data_path when set."""
    base = display_path or data_path
    return os.path.join(base, basename) if os.path.isdir(base) else base


def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    is_wrapped = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = next(iter(state_dict)).startswith("module.")

    if is_wrapped and not has_module_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not is_wrapped and has_module_prefix:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def build_transforms():
    """Preprocessing applied after loading - excludes LoadImaged/EnsureChannelFirstd, which
    NiftiVolumeDataset replaces with a direct nibabel read."""
    return [
        tr.Spacingd(
            keys=["image"],
            pixdim=VOXEL_SPACING,
            mode="bilinear",
        ),
        tr.ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]


class NiftiVolumeDataset:
    """Loads each .nii.gz directly via nibabel instead of monai's LoadImaged+EnsureChannelFirstd,
    whose MetaTensor/meta overhead dominates file I/O for a volume this size."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = nib.load(path)
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        affine = torch.as_tensor(img.affine, dtype=torch.float64)
        volume = MetaTensor(torch.from_numpy(arr)[None], affine=affine, meta={"filename_or_obj": path})
        return self.transform({"image": volume})


def load_data(data_path, transforms_list, num_workers=0):
    images = sorted(glob(os.path.join(data_path, "*.nii.gz")))
    if not images:
        raise FileNotFoundError(f"No .nii.gz files found under: {data_path}")

    # num_workers=0 by default: avoids Docker's default /dev/shm limit crashing worker IPC.
    num_workers = min(num_workers, max(0, len(images) - 1))
    dataset = NiftiVolumeDataset(images, tr.Compose(transforms_list))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return dataloader, images


def inference(args):
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

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

    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    use_amp = bool(args.amp) and str(device).startswith("cuda")

    transforms_list = build_transforms()
    dataloader, image_paths = load_data(args.data_path, transforms_list)

    os.makedirs(args.save_path, exist_ok=True)

    inferer = SlidingWindowInferer(
        roi_size=IMG_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        overlap=SW_OVERLAP,
        mode="gaussian",
    )

    post_transforms_list = [
        tr.Invertd(
            keys="pred",
            transform=tr.Compose(transforms_list),
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        tr.Activationsd(keys="pred", softmax=True),
        tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
    ]
    if args.refine:
        post_transforms_list.append(RefineMaskd(keys="pred"))
    post_transforms_list.append(
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.save_path,
                   separate_folder=False, output_postfix="mask", resample=False)
    )
    post_transforms = tr.Compose(post_transforms_list)

    print("\n" + "=" * 60)
    print("Inference summary")
    print("-" * 60)
    print(f"Input data path : {args.display_path or args.data_path}")
    print(f"Output save path: {args.save_path}")
    print(f"Number of cases : {len(image_paths)} (.nii.gz files)")
    print("=" * 60 + "\n")

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Inference")):
            inputs = data["image"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                data["pred"] = inferer(inputs, model)
            data["pred"] = data["pred"].float()  # back to fp32 for Invertd/SaveImaged
            data = [post_transforms(d) for d in decollate_batch(data)]
            shape = tuple(inputs.shape[2:])
            display_path = display_path_for(os.path.basename(image_paths[i]), args.display_path, args.data_path)
            print(f"[{i + 1}/{len(image_paths)}] predicted: {display_path} (shape={shape})")

    print(f"Done. Predicted masks saved to: {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str,
                         default=str(PROJECT_DIR / "models_ckpt" / "fetbet2_AttUNet3D.pth"),
                         help='path to a trained checkpoint (.pth)')

    parser.add_argument('--data_path', type=str, required=True,
                         help='directory of input .nii.gz volumes to run inference on')

    parser.add_argument('--display_path', type=str, default=None,
                         help='shown as "Input data path" in the summary instead of --data_path '
                              '(e.g. inference.py passes the original path it was given, since '
                              '--data_path here is really just a throwaway symlink directory it '
                              'built); leave unset to show --data_path as-is')

    parser.add_argument('--save_path', type=str, required=True,
                         help='directory to save the predicted masks')

    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu number')

    parser.add_argument('--cpu_threads', type=int, default=None,
                         help='torch CPU thread count override; leave unset for torch\'s own default')
    parser.add_argument('--amp', type=int, default=1,
                         help='use mixed precision (fp16 autocast) on CUDA. Defaults ON: this pipeline\'s '
                              'single large 3D sliding-window pass benefits from Tensor Cores; set to 0 '
                              'to disable if needed.')
    parser.add_argument('--refine', type=int, default=1,
                         help='apply mask_refine.py\'s isotropic-volume cleanup to the predicted '
                              'mask before saving (erosion-based 3D separation of small disconnected '
                              'mis-segmentation from the main brain, since this data has no privileged '
                              'slice axis to chain along). Set to 0 to save the raw predicted mask '
                              'instead. Either way, exactly one mask file is written.')

    parser.add_argument('--device', type=str,
                         default='cuda' if torch.cuda.is_available() else 'cpu',
                         help='device to use')

    args = parser.parse_args()
    inference(args)
