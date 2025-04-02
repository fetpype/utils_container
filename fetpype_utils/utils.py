import SimpleITK as sitk
import numpy as np
from typing import Dict, Any, Optional
import os


def check_same_folder(in_files):
    """
    Check if all the files are in the same folder.
    """
    if len(in_files) > 1:
        assert os.path.commonpath(in_files) == os.path.dirname(
            in_files[0]
        ), f"These files should all be in the same directory: {in_files}"


def check_and_fix_sitk_im_mask(
    image: sitk.Image,
    mask: Optional[sitk.Image],
) -> (sitk.Image, sitk.Image):
    """
    Check if the image and mask are in the same physical space (origin).
    If not, set the mask origin to be the same as the image origin.
    """
    if mask is None:
        return image, None

    if (
        sum([im - m for im, m in zip(image.GetOrigin(), mask.GetOrigin())])
        < 1e-5
    ):
        mask.SetOrigin(image.GetOrigin())
    else:
        raise ValueError(
            "Image and mask are not in the same physical space (origin)."
        )

    if (
        sum([im - m for im, m in zip(image.GetSpacing(), mask.GetSpacing())])
        < 1e-5
    ):
        mask.SetSpacing(image.GetSpacing())
    else:
        raise ValueError(
            "Image and mask are not in the same physical space (spacing)."
        )

    return image, mask


def n4_bias_field_correction_single(
    image: sitk.Image,
    mask: Optional[sitk.Image] = None,
    n4_params: Optional[Dict[str, Any]] = {},
) -> np.ndarray:
    """
    Perform N4 bias field correction on a single image.
    """
    if mask is not None:
        image, mask = check_and_fix_sitk_im_mask(image, mask)
    shrinkFactor = n4_params.get("shrink_factor", 2)
    if shrinkFactor > 1:
        sitk_img = sitk.Shrink(image, [shrinkFactor] * image.GetDimension())
        if mask:
            sitk_mask = sitk.Shrink(
                mask, [shrinkFactor] * image.GetDimension()
            )
    else:

        sitk_img = image
        if mask is not None:
            sitk_mask = mask

    bias_field_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_corrector.SetBiasFieldFullWidthAtHalfMaximum(
        n4_params.get("fwhm", 0.15)
    )
    bias_field_corrector.SetConvergenceThreshold(n4_params.get("tol", 0.001))
    bias_field_corrector.SetSplineOrder(n4_params.get("spline_order", 3))
    bias_field_corrector.SetWienerFilterNoise(n4_params.get("noise", 0.01))
    bias_field_corrector.SetMaximumNumberOfIterations(
        [n4_params.get("n_iter", 59)] * n4_params.get("n_levels", 4)
    )
    bias_field_corrector.SetNumberOfControlPoints(
        n4_params.get("n_control_points", 4)
    )
    bias_field_corrector.SetNumberOfHistogramBins(n4_params.get("n_bins", 200))

    if mask is not None:
        corrected_sitk_img = bias_field_corrector.Execute(sitk_img, sitk_mask)
    else:
        corrected_sitk_img = bias_field_corrector.Execute(sitk_img)

    if shrinkFactor > 1:
        log_bias_field_full = bias_field_corrector.GetLogBiasFieldAsImage(
            image
        )
        corrected_sitk_img_full = image / sitk.Exp(log_bias_field_full)
    else:
        corrected_sitk_img_full = corrected_sitk_img

    corrected_image = sitk.GetArrayFromImage(corrected_sitk_img_full)

    return corrected_image
