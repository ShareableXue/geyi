extern "C" __global__ void fused_add_relu(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float value = a[idx] + b[idx];
    out[idx] = value > 0.0f ? value : 0.0f;
  }
}
