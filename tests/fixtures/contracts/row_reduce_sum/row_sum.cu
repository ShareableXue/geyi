extern "C" __global__ void row_reduce_sum(const float* x, float* out, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float acc = 0.0f;
    for (int col = 0; col < cols; ++col) {
      acc += x[row * cols + col];
    }
    out[row] = acc;
  }
}
