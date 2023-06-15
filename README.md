# Phase-Field-Modelling FNO and PINO

Fourier Neural Operator and Physics-informed Neural Operator method
In this study, we explore the application of the Fourier Neural Operator method (FNO) and Physics-informed Neural Operator method (PINO) to learn the solutions of the Cahn-Morral system of equations. 
FNO and PINO are deep learning methods that are able to learn a resolution-invariant solution operator for a family of time-dependent parametric PDEs

# Requirements:
we used NVIDIA modulus to solve equations.
Requirements can be seen below or in Mosulus Documentation \\
link: https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/getting_started/installation.html

## System Requirements
### Operating System  
Ubuntu 20.04 or Linux 5.13 kernel

### Driver and GPU Requirements
Bare Metal version: NVIDIA driver that is compatible with local PyTorch installation.

### Docker container: 
Modulus container is based on CUDA 11.7, which requires NVIDIA Driver release 515 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 450.51 (or later R450), 470.57 (or later R470), or 510.47 (or later R510). However, any drivers older than 465 will not support the SDF library. For additional support details, see PyTorch NVIDIA Container.

### Required installations for Bare Metal version
Python 3.8
PyTorch 1.12
### Recommended Hardware
64-bit x86
### NVIDIA GPUs:
NVIDIA Ampere GPUs - A100, A30, A4000
Volta GPUs - V100
Turing GPUs - T1

        
