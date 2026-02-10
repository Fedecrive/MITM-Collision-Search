## Compilation
To compile the parallel program `mitm`:

```bash
make
```

### Running on a Cluster (e.g., OAR)
For execution on a cluster with a job scheduler like OAR:

```bash
mpirun -np <num_ranks> --hostfile $OAR_NODEFILE --map-by ppr:1:socket:pe=<num_threads> --bind-to core ./mitm --n <problem_size> --C0 <C0> --C1 <C1> [options]
```

**Arguments for `mitm`:**
- `--n <int>`: Block size in bits (required).
- `--C0 <hex>`: Ciphertext 0 (hex string, required).
- `--C1 <hex>`: Ciphertext 1 (hex string, required).
- `--partitions <int>`: Number of partitions for sliced processing (optional, default 1, needs to be power of 2).
- `--verbose`: Enable verbose output.
