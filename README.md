# MITM Collision Search (Parallel)

This repository contains the implementation of a **parallel Meet-in-the-Middle (MITM) collision search** for the so-called *golden collision* problem, developed as part of the **PPAR course (M2)** at Sorbonne Université.

The project focuses on scaling a given **sequential C implementation** of the MITM attack to handle large input sizes (targeting *n ≥ 40*) by exploiting **parallel programming techniques**.

---

## Project Overview

Starting from a reference sequential implementation, the project introduces parallelism to improve performance and scalability:

- Parallel MITM collision search
- **OpenMP** for shared-memory parallelism
- **MPI** for distributed-memory execution
- Performance and stress testing via custom runner scripts

The goal is to efficiently explore the search space and reduce execution time for large problem instances.

---

## Repository Structure

- `mitm.c` — parallel MITM implementation  
- `mitm_seq.c` — reference sequential version  
- `Makefile` — build configuration  
- `runners/` — scripts for execution, stress tests, and result extraction  
- `Assignment.pdf` — project specification  
- `Final_Project_report.pdf` — final report  
- `Appendix_golden_collisions.pdf` — additional technical material  

---

## Requirements

- C compiler with **OpenMP** support (e.g. `gcc`)
- **MPI** implementation (e.g. OpenMPI or MPICH)
- Unix-like environment (Linux recommended)

---

## Build and Run

The project can be compiled using the provided `Makefile`.  
Execution and benchmarking scripts are available in the `runners/` directory.

(See the final report for implementation details and performance analysis.)

---

## Authors

- **Federico Crivellaro**
- **Gabriele Argentieri**

---

## Academic Information

PPAR — Parallel Programming  
Sorbonne Université (M2)
