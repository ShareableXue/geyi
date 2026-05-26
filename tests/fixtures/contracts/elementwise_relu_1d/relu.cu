extern "C" __global__ void elementwise_relu(const float* x, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
  }
}
