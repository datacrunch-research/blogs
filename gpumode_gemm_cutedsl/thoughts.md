thinking up breaking it up into 2-3 parts, because 1 super long report, no one would read. 

Part 1: 
* Explaining nvfp4 and its relevance, Riccardos previous blog post is very good. Start talking about how Moores law 
* Explaining how data moves on a b200, S2T, T2M etc. 

Part 2: 
* Shortly explaining the advantages of cutedsl and why I chose it
* Creating visual explanation of the software pipeline I made for the gemm problem. 
* Showing code examples from my submission 

Part 3: 
* TBD, maybe talk about how I used nsight to profile kernels. 

Part 1: Why B200 + NVFP4 changes GEMM (Moore’s law pressure, numerics, B200 memory/compute path, S2T/T2M
     fundamentals).
     Part 2: Building the kernel in CuTeDSL (tiling, layouts, TMA/WGMMA pipeline, code walkthrough from minimal to solid
     kernel).
     Part 3: Making it fast and proving it (Nsight workflow, bottlenecks, tuning passes, final perf vs baseline/cuBLAS).

Useful links
https://www.quant.exposed/ from charles_irl
