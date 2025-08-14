# Fetpype -- utils_container
Fetpype `utils_container` is a repository containing tools for the pre-processing of fetal brain MRI T2-weighted stacks prior to super-resolution reconstruction. It is the first step of the main [fetpype](https://fetpype.github.io/fetpype/) pipeline ([github](https://github.com/fetpype/fetpype)). 
These tools are not novel and were not created by us, but this repository merely wraps them in a convenient way to build a container that can be used with Fetpype.

This folder contains the following methods:

- `run_brain_extraction`: Brain extraction using [MONAIfbs](https://github.com/gift-surg/MONAIfbs) (the code was updated to work with more recent MONAI versions) and [fet-bet](https://github.com/IntelligentImaging/fetal-brain-extraction).
- `run_denoising`: Image denoising using ANTS' DenoiseImage.
- `run_bias_field_correction`: Bias field correction using ANTS' implementation of the N4 algorithm.

The main purpose of this repository is to be the base for building the docker container `thsanchez/fetpype_utils:latest` that can be pulled from [docker hub](https://hub.docker.com/r/thsanchez/fetpype_utils) and is required to run `fetpype`. 
