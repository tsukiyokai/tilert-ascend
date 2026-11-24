29: ## Example Description
30: 
31: The GEMV mode refers to the scenario in Mmad computation where M=1, and a left matrix A with shape (1, K) performs matrix multiplication with a right matrix B with shape (K, N). When M=1, GEMV mode is automatically enabled, and it can only be disabled by setting `mmadParams.disableGemv = true` on Ascend 950PR/Ascend 950DT. In this example, the compilation parameter `DISABLE_GEMV` selects whether to disable GEMV mode: 0 means GEMV is enabled, 1 means disabled.
32: 
33: Using M=1, K=256, N=32, with left and right matrix data type half as a specific example, the Mmad computation process in GEMV mode and non-GEMV mode is described below.
34: 
35: - GEMV mode
36: 
37:   When transferring the A matrix from A1 to A2, the 1 * 256 vector is treated as a 16 * 16 matrix, and the LoadData interface is called once to complete the matrix transfer with a 16 * 16 fractal size. The transfer of the B matrix and the matrix multiplication computation are the same as in the basic scenario, as shown in Figure 1 below.
38:   <p align="center">
39:   <img src="figures/开启gemv.png" width="600">
40:   </p>
41:   <p align="center">
42:   Figure 1: GEMV mode, Mmad computation diagram
43:   </p>
44: 
45: - Non-GEMV mode
46: 
47:   When transferring the A matrix from A1 to A2, the 1 * 256 vector is treated as unaligned matrix data, with the M direction aligned to 16 before transfer. The LoadData interface is called to transfer 16 * 16 fractal-sized matrices each time, for a total of CeilDiv(K, 16)=16 transfers, resulting in increased data transfer volume and worse performance compared to GEMV mode, as shown in Figure 2 below.
48:   <p align="center">
49:   <img src="figures/关闭gemv.png" width="600">
50:   </p>
51:   <p align="center">
52:   Figure 2: Non-GEMV mode, Mmad computation diagram
53:   </p>
54: 
55: ## Constraints
56: 
57: - In Mmad computation, to enable GEMV mode, the parameter `mmadParams.m` must equal 1.
58: - In GEMV scenarios, the left matrix A does not support transpose when transferring from L1 to L0A.
59: 
60: ## Build and Run
61: 
