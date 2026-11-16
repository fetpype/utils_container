"""
mask_refine.py

Post-processing for fetal brain-extraction masks: removes spatially disconnected
mis-segmentation (maternal organs, placenta, noise) while staying conservative about
anything that could plausibly be real brain.

Why not naive 3D connected-component filtering: a single-voxel "touch" test lets
contamination merge into the main mask through a sliver of contact; an "area must not
grow between slices" rule fails because real fetal motion causes 2-9x area swings between
adjacent slices; a single-seed-slice heuristic can silently pick the wrong one of two
comparably-sized head positions (if the fetus moved mid-acquisition).

What this module does instead: decomposes each slice into 2D connected components, links
components across slices into chains (requiring real spatial overlap, not mere touching,
within a capped gap), and keeps the largest chain ("the brain"). Any other chain that is
substantial on its own (spans several slices AND is a sizeable fraction of the main chain)
is flagged AMBIGUOUS and left in the output rather than deleted - a real second head
position is worse to lose than suspicious tissue is to flag for a human to check. 4D data
(DWI/fMRI) is refined frame by frame, never chained across frames, since each frame is its
own independently-acquired volume.

Usage
-----
As a library:
    from mask_refine import refine_path
    result = refine_path("sub-01_mask.nii.gz")   # works for 3D or 4D

As a CLI, on a single file or a directory (recursively finds `*_mask_fetal-bet.nii.gz` by
default; pass --suffix to match something else):
    python mask_refine.py /path/to/mask_or_dir [--suffix _mask.nii.gz]

A `<name>_refine.nii.gz` file is written next to each input mask, only when something was
actually removed.
"""
import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

STRUCT2D = ndi.generate_binary_structure(2, 2)


# ---------------------------------------------------------------------------
# Core algorithm (operates on in-memory boolean arrays)
# ---------------------------------------------------------------------------

def detect_slice_axis(zooms, shape):
    """Pick the through-plane (slice) axis: the axis with the largest voxel spacing, or
    (for near-isotropic volumes, where spacing gives no signal) the axis with fewest voxels."""
    zooms = np.asarray(zooms, dtype=float)
    order = np.argsort(-zooms)
    if zooms[order[0]] / zooms[order[1]] > 1.05:
        return int(order[0])
    return int(np.argmin(shape[:3]))


def _slice_components(mask2d):
    lab, n = ndi.label(mask2d, structure=STRUCT2D)
    return [(lab == i) for i in range(1, n + 1)]


def _get_slice(vol, axis, z):
    idx = [slice(None)] * 3
    idx[axis] = z
    return vol[tuple(idx)]


def _overlap_frac(a, b, jitter=1):
    if jitter:
        a = ndi.binary_dilation(a, iterations=jitter)
    inter = np.count_nonzero(a & b)
    denom = min(np.count_nonzero(a), np.count_nonzero(b))
    return inter / denom if denom else 0.0


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def refine_volume(mask, axis, max_gap=2, min_overlap_frac=0.10, jitter=1,
                   min_ambiguous_slices=4, ambiguous_size_frac=0.10,
                   ambiguous_peak_frac=0.30):
    """Refine a single in-memory 3D boolean mask (see module docstring for the algorithm).
    Used directly for a 3D mask, and once per frame for 4D data.

    max_gap: slices at most max_gap+1 apart can still link into the same chain.
    min_overlap_frac: min overlap (intersection / smaller component area) to link two
    components. jitter: dilation (voxels) applied before overlap to tolerate boundary
    jitter. min_ambiguous_slices/ambiguous_size_frac/ambiguous_peak_frac: a non-main chain
    is flagged AMBIGUOUS (kept) instead of auto-removed if it spans enough slices AND is a
    big enough fraction of the main chain's size or peak cross-section.

    Returns a dict: slice_axis, n_slices, orig_voxels, kept_voxels, removed_voxels,
    n_chains, removed_slice_summary, ambiguous_chains, problem (bool), refined (mask)."""
    n = mask.shape[axis]

    empty_result = dict(slice_axis=axis, n_slices=n, orig_voxels=0,
                         kept_voxels=0, removed_voxels=0, n_chains=0,
                         removed_slice_summary=[], ambiguous_chains=[],
                         problem=False, refined=mask)
    if mask.sum() == 0:
        return empty_result

    comps_per_slice = [_slice_components(_get_slice(mask, axis, z)) for z in range(n)]

    node_id = {}
    nodes = []
    for z in range(n):
        for i, c in enumerate(comps_per_slice[z]):
            node_id[(z, i)] = len(nodes)
            nodes.append((z, i, c))

    if not nodes:
        return empty_result

    uf = _UnionFind(len(nodes))
    for za in range(n):
        for zb in range(za + 1, min(n, za + max_gap + 2)):
            for ia, ca in enumerate(comps_per_slice[za]):
                for ib, cb in enumerate(comps_per_slice[zb]):
                    if _overlap_frac(ca, cb, jitter=jitter) >= min_overlap_frac:
                        uf.union(node_id[(za, ia)], node_id[(zb, ib)])

    chains = {}
    for (z, i, c) in nodes:
        root = uf.find(node_id[(z, i)])
        chains.setdefault(root, []).append((z, c))

    chain_stats = []
    for root, members in chains.items():
        total = int(sum(c.sum() for _, c in members))
        zs = sorted(set(z for z, _ in members))
        peak = int(max(c.sum() for _, c in members))
        chain_stats.append(dict(root=root, members=members, total=total,
                                 n_slices=len(zs), z_range=(zs[0], zs[-1]), peak=peak))
    chain_stats.sort(key=lambda d: -d["total"])

    main = chain_stats[0]
    ambiguous = []
    keep_roots = {main["root"]}
    for c in chain_stats[1:]:
        is_ambiguous = (
            c["n_slices"] >= min_ambiguous_slices and
            (c["total"] / main["total"] >= ambiguous_size_frac or
             c["peak"] / main["peak"] >= ambiguous_peak_frac)
        )
        if is_ambiguous:
            keep_roots.add(c["root"])
            ambiguous.append({
                "z_range": c["z_range"], "n_slices": c["n_slices"],
                "voxels": c["total"], "peak_component": c["peak"],
                "size_frac_of_main": c["total"] / main["total"],
                "peak_frac_of_main": c["peak"] / main["peak"],
            })

    accepted = {}
    for cs in chain_stats:
        if cs["root"] not in keep_roots:
            continue
        for z, c in cs["members"]:
            accepted.setdefault(z, np.zeros_like(c))
            accepted[z] |= c

    refined = np.zeros_like(mask)
    for z, comp in accepted.items():
        idx = [slice(None)] * 3
        idx[axis] = z
        refined[tuple(idx)] = comp

    orig_voxels = int(mask.sum())
    kept_voxels = int(refined.sum())
    removed_voxels = orig_voxels - kept_voxels

    removed_slice_summary = []
    for z in range(n):
        raw_area = int(_get_slice(mask, axis, z).sum())
        kept_area = int(accepted[z].sum()) if z in accepted else 0
        removed_here = raw_area - kept_area
        if removed_here > 0:
            removed_slice_summary.append({
                "z": z, "raw_area": raw_area, "kept_area": kept_area,
                "removed": removed_here, "whole_slice_removed": kept_area == 0,
            })

    return dict(
        slice_axis=axis, n_slices=n, orig_voxels=orig_voxels,
        kept_voxels=kept_voxels, removed_voxels=removed_voxels,
        n_chains=len(chain_stats), removed_slice_summary=removed_slice_summary,
        ambiguous_chains=ambiguous,
        problem=(removed_voxels > 0) or bool(ambiguous),
        refined=refined,
    )


def refine_volume_isotropic(mask, erosion_radius=2, ambiguous_size_frac=0.10):
    """Refine a 3D boolean mask for genuinely isotropic volumes (e.g. SVR-reconstructed
    data), which have no privileged slice axis for refine_volume()'s slice-chaining to key
    off. Instead uses marker-based separation: erode by `erosion_radius` voxels (severing
    thin necks while leaving solid structures connected), label the eroded cores in 3D, then
    reassign every voxel of the original mask to its nearest core via a distance transform -
    splitting the mask exactly where a thin bridge was, without needing a slice axis.

    erosion_radius: must be small enough not to erase genuinely thin real structure, large
    enough to sever a plausible contamination neck. ambiguous_size_frac: a non-main core is
    flagged AMBIGUOUS (kept) instead of auto-removed if its size is >= this fraction of the
    main core's.

    Returns a dict: n_cores, orig_voxels, kept_voxels, removed_voxels, ambiguous_chains,
    problem (bool), refined (mask)."""
    orig_voxels = int(mask.sum())
    empty_result = dict(n_cores=0, orig_voxels=orig_voxels, kept_voxels=orig_voxels,
                         removed_voxels=0, ambiguous_chains=[], problem=False, refined=mask)
    if orig_voxels == 0:
        return empty_result

    struct26 = ndi.generate_binary_structure(3, 3)
    eroded = ndi.binary_erosion(mask, iterations=erosion_radius)
    core_labels, n_cores = ndi.label(eroded, structure=struct26)

    if n_cores <= 1:
        # nothing to separate - keep it all rather than lose a thin real structure
        return dict(n_cores=max(n_cores, 1), orig_voxels=orig_voxels, kept_voxels=orig_voxels,
                     removed_voxels=0, ambiguous_chains=[], problem=False, refined=mask)

    # marker-based watershed: assign each voxel to its nearest eroded core
    nearest_idx = ndi.distance_transform_edt(core_labels == 0, return_distances=False, return_indices=True)
    assigned = np.zeros_like(core_labels)
    assigned[mask] = core_labels[tuple(idx[mask] for idx in nearest_idx)]

    sizes = [int(np.count_nonzero(assigned == lbl)) for lbl in range(1, n_cores + 1)]
    order = sorted(range(n_cores), key=lambda i: -sizes[i])
    main_label = order[0] + 1
    main_size = sizes[order[0]]

    keep_labels = {main_label}
    ambiguous = []
    for i in order[1:]:
        label, size = i + 1, sizes[i]
        if size / main_size >= ambiguous_size_frac:
            keep_labels.add(label)
            ambiguous.append({"voxels": size, "size_frac_of_main": size / main_size})

    refined = np.isin(assigned, list(keep_labels))
    kept_voxels = int(refined.sum())
    removed_voxels = orig_voxels - kept_voxels

    return dict(
        n_cores=n_cores, orig_voxels=orig_voxels, kept_voxels=kept_voxels,
        removed_voxels=removed_voxels, ambiguous_chains=ambiguous,
        problem=(removed_voxels > 0) or bool(ambiguous), refined=refined,
    )


# ---------------------------------------------------------------------------
# File-level wrappers
# ---------------------------------------------------------------------------

def refine_mask_file(mask_path, **kwargs):
    """Refine a 3D mask NIfTI file. Returns the refine_volume() result dict plus `path`,
    `affine`, `header`."""
    img = nib.load(mask_path)
    mask = np.asarray(img.dataobj) > 0
    axis = detect_slice_axis(img.header.get_zooms()[:3], mask.shape)
    r = refine_volume(mask, axis, **kwargs)
    r.update(path=mask_path, affine=img.affine, header=img.header)
    return r


def refine_mask_file_4d(mask_path, **kwargs):
    """Refine a 4D mask NIfTI file (e.g. DWI/fMRI) frame by frame, independently (no
    chaining across frames). Returns a dict: path, affine, header, slice_axis,
    spatial_shape, n_frames, orig_voxels, kept_voxels, removed_voxels, n_flagged_frames,
    frame_reports (per-frame results, each tagged with frame index `t`), refined
    ((X, Y, Z, T) uint8), problem (bool)."""
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj) > 0
    if data.ndim != 4:
        raise ValueError(f"expected a 4D mask, got shape {data.shape}")

    spatial_shape = data.shape[:3]
    n_frames = data.shape[3]
    axis = detect_slice_axis(img.header.get_zooms()[:3], spatial_shape)

    refined4d = np.zeros_like(data, dtype=np.uint8)
    frame_reports = []
    orig_voxels = 0
    removed_voxels = 0
    n_flagged_frames = 0
    for t in range(n_frames):
        r = refine_volume(data[..., t], axis, **kwargs)
        refined4d[..., t] = r["refined"]
        orig_voxels += r["orig_voxels"]
        removed_voxels += r["removed_voxels"]
        if r["problem"]:
            n_flagged_frames += 1
        r["t"] = t
        frame_reports.append(r)

    return dict(
        path=mask_path, affine=img.affine, header=img.header,
        slice_axis=axis, spatial_shape=spatial_shape, n_frames=n_frames,
        orig_voxels=orig_voxels, kept_voxels=orig_voxels - removed_voxels,
        removed_voxels=removed_voxels, n_flagged_frames=n_flagged_frames,
        frame_reports=frame_reports,
        problem=any(r["problem"] for r in frame_reports),
        refined=refined4d,
    )


def refine_path(mask_path, **kwargs):
    """Refine a mask file, dispatching to the 3D or 4D routine based on its dimensionality."""
    ndim = len(nib.load(mask_path).shape)
    if ndim == 3:
        return refine_mask_file(mask_path, **kwargs)
    if ndim == 4:
        return refine_mask_file_4d(mask_path, **kwargs)
    raise ValueError(f"unsupported mask dimensionality {ndim} for {mask_path}")


def save_refined(result, mask_path, out_suffix="_refine.nii.gz", in_suffix=".nii.gz"):
    """Write `result['refined']` next to `mask_path`, only if voxels were actually removed
    (also removes a stale output from a previous run otherwise). Returns the output path,
    or None."""
    if mask_path.endswith(in_suffix):
        out_path = mask_path[: -len(in_suffix)] + out_suffix
    elif mask_path.endswith(".nii.gz"):
        out_path = mask_path[: -len(".nii.gz")] + out_suffix
    else:
        out_path = os.path.splitext(mask_path)[0] + out_suffix

    if result["removed_voxels"] > 0:
        out_img = nib.Nifti1Image(result["refined"].astype(np.uint8),
                                   result["affine"], result["header"])
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, out_path)
        return out_path

    if os.path.exists(out_path):
        os.remove(out_path)
    return None


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

def _print_report(rel, result):
    if "frame_reports" in result:  # 4D
        pct = 100.0 * result["removed_voxels"] / result["orig_voxels"] if result["orig_voxels"] else 0.0
        n_amb = sum(len(fr["ambiguous_chains"]) for fr in result["frame_reports"])
        print(f"FLAG {rel}  frames_flagged={result['n_flagged_frames']}/{result['n_frames']} "
              f"removed={result['removed_voxels']}/{result['orig_voxels']} ({pct:.2f}%) "
              f"ambiguous_instances={n_amb}")
        for fr in result["frame_reports"]:
            if not fr["problem"]:
                continue
            fpct = 100.0 * fr["removed_voxels"] / fr["orig_voxels"] if fr["orig_voxels"] else 0.0
            print(f"    frame t={fr['t']:3d} removed={fr['removed_voxels']}/{fr['orig_voxels']} ({fpct:.1f}%)")
            for a in fr["ambiguous_chains"]:
                print(f"        AMBIGUOUS chain kept z={a['z_range']} n_slices={a['n_slices']} "
                      f"voxels={a['voxels']} ({a['size_frac_of_main']*100:.0f}% of main, "
                      f"{a['peak_frac_of_main']*100:.0f}% of peak)")
    else:  # 3D
        pct = 100.0 * result["removed_voxels"] / result["orig_voxels"] if result["orig_voxels"] else 0.0
        whole = sum(1 for s in result["removed_slice_summary"] if s["whole_slice_removed"])
        print(f"FLAG {rel}  removed={result['removed_voxels']}/{result['orig_voxels']} "
              f"({pct:.2f}%) whole_slices={whole}")
        for s in result["removed_slice_summary"]:
            tag = "WHOLE" if s["whole_slice_removed"] else "partial"
            print(f"    z={s['z']:3d} raw={s['raw_area']:5d} kept={s['kept_area']:5d} "
                  f"removed={s['removed']:5d} [{tag}]")
        for a in result["ambiguous_chains"]:
            print(f"    AMBIGUOUS chain kept z={a['z_range']} n_slices={a['n_slices']} "
                  f"voxels={a['voxels']} ({a['size_frac_of_main']*100:.0f}% of main, "
                  f"{a['peak_frac_of_main']*100:.0f}% of peak)")


#: known mask filename suffixes, tried in order to infer what to replace with "_refine.nii.gz"
KNOWN_MASK_SUFFIXES = ["_mask_fetal-bet.nii.gz", "_mask.nii.gz"]


def _infer_in_suffix(path, preferred):
    for s in [preferred] + [s for s in KNOWN_MASK_SUFFIXES if s != preferred]:
        if path.endswith(s):
            return s
    return ".nii.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a single mask file, or a directory to search recursively")
    ap.add_argument("--suffix", default="_mask_fetal-bet.nii.gz",
                     help="filename suffix to match when `path` is a directory "
                          "(default: _mask_fetal-bet.nii.gz)")
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    if os.path.isfile(args.path):
        mask_paths = [args.path]
        root = os.path.dirname(args.path) or "."
        print(f"Processing single file: {args.path}\n")
    else:
        out_suffix = args.suffix.replace(".nii.gz", "_refine.nii.gz")
        mask_paths = sorted(glob.glob(os.path.join(args.path, "**", f"*{args.suffix}"), recursive=True))
        mask_paths = [p for p in mask_paths if not p.endswith(out_suffix)]
        root = args.path
        print(f"Found {len(mask_paths)} mask file(s) under {root} matching *{args.suffix}\n")

    n_flagged = 0
    n_errors = 0
    for idx, p in enumerate(mask_paths):
        rel = os.path.relpath(p, root)
        try:
            result = refine_path(p, max_gap=args.max_gap)
        except Exception as e:
            print(f"[{idx+1}/{len(mask_paths)}] ERROR {rel}: {e}")
            traceback.print_exc()
            n_errors += 1
            continue

        if result["problem"]:
            n_flagged += 1
            print(f"[{idx+1}/{len(mask_paths)}] ", end="")
            _print_report(rel, result)
        else:
            print(f"[{idx+1}/{len(mask_paths)}] clean {rel}")

        if not args.dry_run:
            in_suffix = _infer_in_suffix(p, args.suffix)
            out_suffix = in_suffix.replace(".nii.gz", "_refine.nii.gz")
            save_refined(result, p, out_suffix=out_suffix, in_suffix=in_suffix)

    print(f"\n{n_flagged} / {len(mask_paths)} masks flagged, {n_errors} error(s)")


if __name__ == "__main__":
    main()
