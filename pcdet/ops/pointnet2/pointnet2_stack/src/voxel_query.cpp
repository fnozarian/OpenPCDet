#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "voxel_query_gpu.h"

#include <torch/types.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)



int voxel_query_wrapper_stack(int M, int R1, int R2, int R3, int nsample, float radius, 
    int z_range, int y_range, int x_range, at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, 
    at::Tensor new_coords_tensor, at::Tensor point_indices_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_coords_tensor);
    CHECK_INPUT(point_indices_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    const int *new_coords = new_coords_tensor.data<int>();
    const int *point_indices = point_indices_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    voxel_query_kernel_launcher_stack(M, R1, R2, R3, nsample, radius, z_range, y_range, x_range, new_xyz, xyz, new_coords, point_indices, idx);
    return 1;
}
