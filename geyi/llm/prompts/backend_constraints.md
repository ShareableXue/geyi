Allowed Phase 2 backend targets:
- tilelang.fused_add_relu_1d for 1D contiguous elementwise fused add followed by relu.
- existing deterministic templates may be selected only when they match the contract intent.
Generated code must still be compiled and verified by Geyi.
