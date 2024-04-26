# Description
This repository contains the code and data for simulating the Generalized XY (GXY) Model in two Euclidean dimensions using Tensor Renormalization Group (TRG) technique. We are using HOTRG (Higher Order Tensor Renormalization Group) method to study the phase structure. It is also available at https://zenodo.org/record/10999416


The Hamiltonian for this model is given by,
$$\mathcal{H} = -\left[\Delta \sum_{\langle j k \rangle} \cos{(\theta_j - \theta_k)} + (1-\Delta) \sum_{\langle j k \rangle} \cos{(q(\theta_j - \theta_k))} + h \sum_j \cos{(\theta_j)} + h_1\sum_j \cos{(q\theta_j)}\right]$$

where, $\Delta \in [0,1]$ is called the deformation parameter. $h$ and $h_1$ are symmetry breaking external magnetic fields. We study the model for $q = 2$ case.

# Contents
The file ```main.py``` contains the routines to use HOTRG algorithm for tensor networks study of the GXY model.

```data``` directory contains all the simulation data for the article : https://www.arxiv.org/24xx.xxxxx

# Usage
The file ```main.py``` requires the following six input parameters,
1. ```T```     : Temperature
2. ```h```     : Value of external magnetic field $h$
3. ```h1```    : Value of external magnetic field $h_1$
4. ```D```     : Value of the bond dimension to be used for HOTRG algorithm
5. ```N```     : Number of iterations to be used for HOTRG algorithm
6. ```Delta``` : Value of the deformation parameter $\Delta$ for the 2d GXY model

## Dependencies
1. ```opt_einsum```
2. ```PyTorch```
3. ```SciPy```
4. ```NumPy```

## Architecture
The code can run on both ```CPU``` and ```NVIDIA GPU``` based on their availability. We recommend using it with ```GPUs``` to get numerical results quickly. The boolean variable ```use_cuda``` can be manually set to ```True``` or ```False``` to enfore the program to run over a specific architecture. For further details on the GPU acceleration for TRG methods, see https://arxiv.org/abs/2306.00358

## Command
Use the following command to run the python code,

```python3 main.py T h h1 D N Delta```

## Cite our paper
We request you to cite our paper if you find this repository helpful.  
