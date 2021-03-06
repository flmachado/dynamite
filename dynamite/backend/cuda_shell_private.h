
#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <petscmat.h>
#include <petsccuda.h>
#include "shellcontext.h"

#define GPU_BLOCK_SIZE 128
#define GPU_BLOCK_NUM 128

__global__ void device_MatMult_Shell(PetscInt size,
                                     PetscInt* masks,
                                     PetscInt* signs,
                                     PetscScalar* coeffs,
                                     PetscInt nterms,
                                     const PetscScalar* xarray,
                                     PetscScalar* barray);

__global__ void device_MatNorm_Shell(PetscInt size,
                                     PetscInt* masks,
                                     PetscInt* signs,
                                     PetscScalar* coeffs,
                                     PetscInt nterms,
                                     PetscReal* d_maxs);

