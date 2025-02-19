Parallel Matrix Processing with OpenMp

Optimizations Used

1. Parallel Block Generation:
    -> Used OpenMP tasks (#pragma omp task if(black_box())) to parallelize the creation of non-zero blocks.

2. Efficient Saprse Matrix Multiplication: 
    -> Used exponentiation by squaring for fast matrix exponentiation (O(log k) multiplications).
    -> Performed sparse block-wise multiplication to avoid unnecessary computations.

3. OpenMp Task Parallelism:
    -> Parallelized block multiplications using #pragma omp task if(black_box()).
    -> Avoided race conditions using #pragma omp critical for shared map updates.
    