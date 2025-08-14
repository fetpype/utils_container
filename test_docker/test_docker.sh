# Create folders docker_masks, docker_denoise, docker_debias

mkdir -p docker_masks
mkdir -p docker_denoise
mkdir -p docker_debias

# Create a variable test that contains
# Get it as a variancel /home/tsanchez/Documents/mial/repositories/fetpype_utils/test
in_dir="/home/tsanchez/Documents/mial/repositories/fetpype_utils/"

# Same but with GPU
docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_masks:/masks --rm fetpype/utils:latest run_brain_extraction --input_dir /data --output_dir /masks --method fet_bet
docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_masks:/masks --rm fetpype/utils:latest run_brain_extraction --input_dir /data --output_dir /masks --method monaifbs

# Test denoising
docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_denoise:/denoise --rm fetpype/utils:latest run_denoising --input_stacks /data/sub-chuv014_ses-01_acq-haste_run-1_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-3_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-4_T2w.nii.gz --output_stacks /denoise/sub-chuv014_ses-01_acq-haste_run-1_T2w.nii.gz /denoise/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /denoise/sub-chuv014_ses-01_acq-haste_run-3_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-4_T2w.nii.gz

# Test biasfield correction
docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_debias:/debias --rm fetpype/utils:latest run_bias_field_correction --input_stacks /data/sub-chuv014_ses-01_acq-haste_run-1_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-3_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-4_T2w.nii.gz --output_stacks /debias/sub-chuv014_ses-01_acq-haste_run-1_T2w.nii.gz /debias/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /debias/sub-chuv014_ses-01_acq-haste_run-3_T2w.nii.gz /debias/sub-chuv014_ses-01_acq-haste_run-4_T2w.nii.gz 



### Testing singularity version
docker run --gpus all --rm -v "$PWD":/data gerardmartijuan/fetpype_utils:latest run_brain_extraction --input_stacks /data/sub-CC00864XX15_ses-2731_run-09_T2w.nii.gz /data/sub-CC00864XX15_ses-2731_run-10_T2w.nii.gz /data/sub-CC00864XX15_ses-2731_run-11_T2w.nii.gz --output_masks /data/sub-CC00864XX15_ses-2731_run-09_T2w_mask.nii.gz /data/sub-CC00864XX15_ses-2731_run-10_T2w_mask.nii.gz /data/sub-CC00864XX15_ses-2731_run-11_T2w_mask.nii.gz --method fet_bet


docker run --gpus all --rm -v "$PWD":/data gerardmartijuan/fetpype_utils:latest run_brain_extraction --input_dir /data/data --output_dir /data/masks --method fet_bet


docker run --gpus all --rm -v "$PWD":/data thsanchez/fetpype_utils:latest run_brain_extraction --input_stacks /data/sub-CC00864XX15_ses-2731_run-13_T2w.nii.gz --output_masks /data/sub-CC00864XX15_ses-2731_run-13_T2w_mask.nii.gz --method monaifbs
