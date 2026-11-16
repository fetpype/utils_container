from glob import glob
from pathlib import Path

import numpy as np
import argparse
import os

import torch

import monai.transforms as tr
from monai.data import MetaTensor, decollate_batch, Dataset, DataLoader
from monai.inferers import SliceInferer
from monai.networks.nets import AttentionUnet
from monai.transforms import SaveImaged, MapTransform
from tqdm import tqdm

from mask_refine import detect_slice_axis, refine_volume

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class RefineMaskd(MapTransform):
    """Cleans up small disconnected mis-segmented components in the predicted mask (see
    mask_refine.py), in memory before SaveImaged."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            pred = d[key]
            arr = pred.as_tensor() if hasattr(pred, "as_tensor") else pred
            mask = arr[0].cpu().numpy() > 0  # drop channel dim -> (H, W, D)
            zooms = np.linalg.norm(np.asarray(pred.affine)[:3, :3], axis=0)
            axis = detect_slice_axis(zooms, mask.shape)
            refined = refine_volume(mask, axis)["refined"]
            new_data = torch.from_numpy(refined).to(arr.dtype)[None]
            d[key] = MetaTensor(new_data, meta=pred.meta) if hasattr(pred, "meta") else new_data
        return d

class SliceWiseNormalizeIntensityd(MapTransform):
    """Per-slice intensity normalization: (slice - subtrahend) / divisor, defaulting to the
    slice's own mean/std. Vectorized across slices instead of a Python loop (~30x faster)."""
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            x = image.as_tensor().clone() if hasattr(image, "as_tensor") else image.clone()
            reduce_dims = tuple(range(x.dim() - 1))  # everything except the last (slice) dim

            if self.nonzero:
                mask = x > 0
                n = mask.sum(dim=reduce_dims, keepdim=True).clamp(min=1).to(x.dtype)
                nan_x = torch.where(mask, x, torch.full_like(x, float("nan")))
                # std is always computed around the true mean, independent of `subtrahend`
                true_mean = torch.nanmean(nan_x, dim=reduce_dims, keepdim=True)
                if self.divisor is None:
                    var = torch.nanmean((nan_x - true_mean) ** 2, dim=reduce_dims, keepdim=True)
                    var = var * n / (n - 1).clamp(min=1)  # Bessel's correction, matches tensor.std()
                    div = torch.sqrt(var)
                else:
                    div = self.divisor
                sub = true_mean if self.subtrahend is None else self.subtrahend
                centered = x - sub
                x = torch.where(mask, centered / div, x)
            else:
                sub = x.mean(dim=reduce_dims, keepdim=True) if self.subtrahend is None else self.subtrahend
                centered = x - sub
                div = centered.std(dim=reduce_dims, keepdim=True) if self.divisor is None else self.divisor
                x = centered / div

            if hasattr(image, "copy_"):
                image.copy_(x)
            else:
                image = x
            d[key] = image
        return d


class FetalTestData:
    def __init__(self, test_data_paths, img_size=256):
        self.test_data_paths = test_data_paths
        self.img_size = img_size

    def transformations(self):
        test_transforms_list = [
            tr.LoadImaged(keys=["image"], image_only=True),
            tr.EnsureChannelFirstd(keys=["image"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear", padding_mode="zeros"),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
        ]
        return test_transforms_list

    def load_data(self):
        test_transforms_list = self.transformations()
        test_images = sorted(glob(os.path.join(self.test_data_paths, "*.nii.gz")))

        test_files = [{"image": image_name} for image_name in test_images]

        test_dataset = Dataset(data=test_files, transform=tr.Compose(test_transforms_list))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     num_workers=0)

        return test_dataloader, test_transforms_list, test_images


def predict_with_flip_tta(inferer, model, test_inputs):
    """4-way flip TTA (identity + H/W/HW flips). Only the in-plane axes are flipped; the
    slice-selection axis is untouched."""
    pred_orig = inferer(test_inputs, model)

    flip_inputs_h = torch.flip(test_inputs, dims=(2,))
    pred_h = torch.flip(inferer(flip_inputs_h, model), dims=(2,))

    flip_inputs_w = torch.flip(test_inputs, dims=(3,))
    pred_w = torch.flip(inferer(flip_inputs_w, model), dims=(3,))

    flip_inputs_hw = torch.flip(test_inputs, dims=(2, 3))
    pred_hw = torch.flip(inferer(flip_inputs_hw, model), dims=(2, 3))

    return (pred_orig + pred_h + pred_w + pred_hw) / 4


def display_path_for(basename, display_path, data_path):
    """Map a file's basename back to a user-facing path for printing, using --display_path
    (the caller's real path) instead of this script's own --data_path when set."""
    base = display_path or data_path
    return os.path.join(base, basename) if os.path.isdir(base) else base


def load_checkpoint(model, checkpoint_path, device):
    """Load a checkpoint regardless of whether it (or the model) is DataParallel-wrapped, by
    matching the "module." prefix on both sides."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    is_wrapped = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = next(iter(state_dict)).startswith("module.")

    if is_wrapped and not has_module_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not is_wrapped and has_module_prefix:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


def inference(args):
    torch.set_num_threads(args.cpu_threads)

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(64, 128, 256, 512, 1024),
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

    use_amp = bool(args.amp) and str(device).startswith("cuda")

    fetal_test_data = FetalTestData(args.data_path)
    test_dataloader, test_org_transforms_list, image_paths = fetal_test_data.load_data()

    inferer = SliceInferer(
        roi_size=(224, 224),
        spatial_dim=2,
        sw_batch_size=4,
        overlap=0.25,
        mode="gaussian",
        progress=False
    )

    post_transforms_list = [
        tr.Invertd(
            keys="pred",
            transform=tr.Compose(test_org_transforms_list),
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
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.save_path, print_log=False,
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
        for i, test_data in enumerate(tqdm(test_dataloader, desc="Inference")):
            test_inputs = test_data["image"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                test_data["pred"] = predict_with_flip_tta(inferer, model, test_inputs)
            test_data["pred"] = test_data["pred"].float()  # back to fp32 for Invertd/SaveImaged
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            shape = tuple(test_inputs.shape[2:])
            display_path = display_path_for(os.path.basename(image_paths[i]), args.display_path, args.data_path)
            print(f"[{i + 1}/{len(image_paths)}] predicted: {display_path} (shape={shape})")

    print(f"Done. Predicted masks saved to: {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_gpu',
                        type=int,
                        default=1,
                        help='total gpu number (wraps the model in DataParallel if > 1). The '
                             'checkpoint loads correctly either way now (load_checkpoint matches '
                             'the "module." prefix automatically), so this is purely about whether '
                             'you want multi-GPU parallelism, not a required value to guess.')

    parser.add_argument('--cpu_threads',
                        type=int,
                        default=4,
                        help='torch CPU thread count for preprocessing/postprocessing. Lower is '
                             'faster here (this pipeline\'s tensors are small - thread scheduling '
                             'overhead dominates past a handful of threads); raise it if you run '
                             'much larger volumes through this script.')

    parser.add_argument('--refine',
                        type=int,
                        default=1,
                        help='apply mask_refine.py\'s connected-component cleanup to the predicted '
                             'mask before saving (removes small disconnected mis-segmentation while '
                             'keeping ambiguous competing candidates). Set to 0 to save the raw '
                             'predicted mask instead. Either way, exactly one mask file is written.')

    parser.add_argument('--amp',
                        type=int,
                        default=0,
                        help='use mixed precision (fp16 autocast) on CUDA. Defaults OFF here: for this '
                             '2D + 4-way-TTA pipeline (many small 2D conv calls) autocast overhead can '
                             'outweigh the fp16 compute savings, unlike inference_3d.py\'s single large '
                             '3D pass. Try --amp 1 if your GPU/data differ enough that this tradeoff flips.')

    parser.add_argument('--device',
                        type=str,
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='what device to use')

    parser.add_argument('--saved_model_path',
                        type=str,
                        default=str(PROJECT_DIR / "models_ckpt" / "fetbet2_AttUNet2D.pth"),
                        help='path to the saved model')

    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to the test data')

    parser.add_argument('--display_path',
                        type=str,
                        default=None,
                        help='shown as "Input data path" in the summary instead of --data_path '
                             '(e.g. inference.py passes the original path it was given, since '
                             '--data_path here is really just a throwaway symlink directory it '
                             'built); leave unset to show --data_path as-is')

    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help='path to save the out')

    args = parser.parse_args()
    inference(args)
