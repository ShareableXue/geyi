extern "C" __global__ void inline_ptx_add(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    asm volatile("");
    out[idx] = a[idx] + b[idx];
  }
}

