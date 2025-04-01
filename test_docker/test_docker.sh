# Create folders docker_masks, docker_denoise, docker_debias

mkdir -p docker_masks
mkdir -p docker_denoise
mkdir -p docker_debias

# Create a variable test that contains
# Get it as a variancel /home/tsanchez/Documents/mial/repositories/fetpype_utils/test
in_dir="/home/tsanchez/Documents/mial/repositories/fetpype_utils/"

docker run -v ${in_dir}test:/data -v ${in_dir}docker_masks:/masks --rm fetpype/utils:latest run_brain_extraction --input_dir /data --out_dir /masks --method fet_bet --device cpu
docker run -v ${in_dir}test:/data -v ${in_dir}docker_masks:/masks --rm fetpype/utils:latest run_brain_extraction --input_dir /data --out_dir /masks --method monaifbs --device cpu
# Same but with GPU
docker run --gpus all -v ${in_dir}:/data -v ${in_dir}docker_masks:/masks --rm fetpype/utils:latest run_brain_extraction --input_dir /data --out_dir /masks --method fet_bet
docker run --gpus all -v ${in_dir}:/data -v ${in_dir}docker_masks:/masks --rm fetpype/utils:latest run_brain_extraction --input_dir /data --out_dir /masks --method monaifbs

# Test denoising
docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_denoise:/denoise --rm fetpype/utils:latest run_denoising --input_stacks /data/sub-chuv014_ses-01_acq-haste_run-1_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-3_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-4_T2w.nii.gz --output_dir /denoise 

# Test biasfield correction
docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_debias:/debias --rm fetpype/utils:latest run_bias_field_correction --input_stacks /data/sub-chuv014_ses-01_acq-haste_run-1_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-3_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-4_T2w.nii.gz --output_dir /debias 


docker run --gpus all -v ${in_dir}test:/data -v ${in_dir}docker_debias:/debias --rm fetpype/utils:latest run_bias_field_correction --input_stacks /data/sub-chuv083_ses-01_acq-haste_run-1_T2w.nii.gz /data/sub-chuv014_ses-01_acq-haste_run-2_T2w.nii.gz /data/sub-chuv083_ses-01_acq-haste_run-3_T2w.nii.gz /data/sub-chuv083_ses-01_acq-haste_run-4_T2w.nii.gz --output_dir /debias 
