extern "C" __global__ void cast1d(const int* x, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = (float)x[idx];
  }
}
