# Storage-preserving INT32 NZ alias

This independent host extension corrects the physical descriptor of the supported W4A8 GU/Down carriers. It creates a new `StorageImpl` over the same allocation and retains the original Tensor in its deleter. It does not modify the original descriptor, raw bytes, logical tensor geometry, model arithmetic, or official operators.

The [build manifest](build/manifest.json) records the Torch/TorchNPU 2.12 ABI, exact compiler and linker commands, image identity, dependency hashes, loaded schemas and absence of NPU initialization. The library SHA256 is `d420080a55500e4d908161bcd23b6bf92f4a5c9ee8d8f0d3020a1ba2a2a9ad4a`. Binaries remain at the paths in [binary-artifacts.json](binary-artifacts.json).

The [descriptor audit and device gates](../decoder-nz-descriptor-audit-v1/README.md) distinguish the original address failure, corrected synchronized execution, original asynchronous execution, and the later independent attention precision failure. The additional HBM allocation is zero; each independent descriptor keeps the existing allocation alive.
