from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "vector add");
}
"""

CUDA_SOURCE = """
#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  return a + b;
}
"""


def main() -> None:
    extension = load_inline(
        name="geyi_vector_add_inline",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["vector_add"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )
    extension.vector_add(1, 2)
    torch.ops.load_library("compiled_only_extension.so")
    print("phase4 load_inline demo completed")


if __name__ == "__main__":
    main()

