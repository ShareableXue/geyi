extern "C" __global__ void transpose2d(const float* x, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    out[col * rows + row] = x[row * cols + col];
  }
}
