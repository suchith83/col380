# COL380 Assignment 4 Report

## Name: Koduru Suchith  
## Entry Number: <Your Entry Number Here>

---

## Parallelization Strategy

### MPI Distribution
- Input: N sparse matrices split among `m` MPI processes.
- Each process is assigned roughly `N/m` matrices.
- Local multiplication is done in-process, followed by a **binary tree reduction** to combine results.

### Intra-Node Parallelism: OpenMP
- Used for parallel loading of blocks and CPU-side multiplication of sparse matrices.
- Critical sections are used to manage concurrent updates to the sparse result matrix.

### GPU Acceleration: CUDA
- CUDA is used to perform multiplication of individual dense `k x k` blocks.
- Each multiplication is launched as a CUDA kernel using `blockMultiplyKernel`.

---

## Performance Discussion

### Test Setup
- Matrices of size: `4096 x 4096`
- Block size `k`: 16
- Number of matrices: 50
- Nodes: 4 MPI processes
- Each with 1 GPU and 16 OpenMP threads

### Observations
- CUDA acceleration significantly improves per-block multiplication time.
- MPI reduction (tree-based) effectively reduces overall communication cost.
- OpenMP allows concurrent parsing and CPU multiplication.
- Final result correctness verified by test harness.

---

## Notes
- Input and output formats follow exact block-matrix specification.
- Output matrix is saved as `./matrix`, not inside input folder (read-only assumption).
- Efficiency can be further improved with better scheduling heuristics (e.g., greedy matrix order).

---
