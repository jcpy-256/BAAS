# BAAS 
BAAS, a high-performance framework designed for parallelizing sparse matrix computations. This work is featured in our IPDPS 2026 paper: “BAAS: A Bidirectional Aggregation and Affinity-Aware Scheduling Framework for Parallelizing Sparse Matrix Computations.”

## Prerequisites
* **Intel MKL**: For optimized BLAS/Sparse kernels.

* **Metis**: For graph partitioning and matrix reordering.

* **PAPI**: For hardware performance counter monitoring and profiling.

* **CMake**: Version 3.10 or higher

* **GCC**: With C++17 support and OpenMP.

## Getting Started
**1. Configuration**
First, you need to point BAAS to your local library installations.
Navigate to the `script` directory and edit `env.sh`:
```shell
cd script
# Open env.sh and update the paths for MKL, Metis, and PAPI
vi env.sh
```

**2. Build**

We provide convenient scripts to handle the build process:
```shell 
# Generate Build files
bash cmake.sh

# Compile the project
bash make.sh
```
**3. Data Preparation**
BAAS expects matrix files to be located in the /data directory.

* Place your .mtx files in `/data`.

* Update `matrixList.txt` to include the filenames of the matrices you wish to test.

**4. Running**
We have provided specialized scripts for different sparse solvers. You can run them directly from the script directory.
```shell
./run_trsv.sh # SpTRSV 
./run_ic.sh   # SpIC0
./run_ilu.sh  # SpILU0
./run_iccg.sh # ICCG
```
**Note**: You can customize the **thread count** and **iteration rounds** within these scripts to match your hardware.

**5. Output & Results:**
All generated files are stored in the output/ directory:

`output/csv/`: Contains the measured performance metrics and benchmark results.

`output/log/`: Contains detailed program execution logs.

`output/out/`: Contains standard output redirected from the running scripts.