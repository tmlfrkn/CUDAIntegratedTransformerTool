# CUDA Integrated Transformer Tool

This tool is used to modify and transform CUDA source codes. This README file provides instructions on how to install, configure, and use the tool.

## Installation

1. To use this tool, first download and compile the [LLVM project](https://llvm.org/).

2. While compiling the LLVM project, create a subdirectory under the `clang-tools-extra` directory named `CUDAIntegratedTransformerTool`.

3. Add the following line to the `CMakeLists.txt` file under the `clang-tools-extra` directory:
    ```cmake
    add_subdirectory(CUDAIntegratedTransformerTool)
    ```

4. Create a `CMakeLists.txt` file under the `CUDAIntegratedTransformerTool` directory and add the following contents:
    ```cmake
    set(LLVM_LINK_COMPONENTS support)
    add_clang_executable(CUDAIntegratedTransformerTool
        CUDAIntegratedTransformerTool.cpp
    )
    target_link_libraries(CUDAIntegratedTransformerTool
        PRIVATE
        clangAST
        clangASTMatchers
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling
        clangEdit
    )
    ```

5. Place the source code for the tool under the `CUDAIntegratedTransformerTool` directory.

6. Finally, navigate to the build directory of the LLVM project and compile the tool by running the following commands:
    ```bash
    cd /path/to/llvm-project/build
    make
    ```

## Usage

When using the tool, you can utilize the following command line options:

- `--threads=<value>`: Specifies the number of threads for modification. Example usage: `--threads=128`.
- `--reduction-ratio=<value>`: Specifies the reduction ratio for the number of threads. Example usage: `--reduction-ratio=50`.
- `--convert-double-to-float`: Converts double types to float types. Example usage: `--convert-double-to-float`.
- `--change-Kernel`: Allows changing the parameters of CUDA kernel launch statements. Example usage: `--change-Kernel`.
- `--kernelParam-num=<value>`: Specifies which parameter of the kernel launch statement to modify (1 or 2). Example usage: `--kernelParam-num=1`.
- `--dim3`: Allows changing dim3 declaration parameters. Example usage: `--dim3`.
- `--num-dim3-changes=<number>`: Specifies the number of dim3 declarations to change. Example usage: `--num-dim3-changes=2`.
- `--change-var-name=<variable_name>`: Specifies the name of the variable to be changed. Example usage: `--change-var-name=THREADS`.
- `--change-specific`: Allows changing the value of a specific variable. Example usage: `--change-specific`.
- `--remove_synch_thread_to_null`: Replaces __syncthreads() function calls with "NULL()". Example usage: `--remove_synch_thread_to_null`.
- `--remove_synch_thread_to_empty`: Replaces __syncthreads() function calls with an empty string. Example usage: `--remove_synch_thread_to_empty`.
- `--replace-with-syncwarp`: Replaces __syncthreads() function calls with __syncwarp(). Example usage: `--replace-with-syncwarp`.
- `--atomic-add-to-atomic-add-block`: Replaces atomicAdd() function calls with atomicAddBlock(). Example usage: `--atomic-add-to-atomic-add-block`.
- `--atomic-to-direct`: Replaces atomicAdd() function calls with direct operations. Example usage: `--atomic-to-direct`.
- `--simplify-if-statements`: Simplifies function bodies by keeping only the first if statement body. Example usage: `--simplify-if-statements`.


Example command line executions:

```bash
CUDAIntegratedTransformerTool --convert-if-else-to-if-body=true  /home/furkan/Desktop/PolyBench/PolyBench-ACC-master/CUDA/datamining/correlation/correlation.cu -- -I/home/furkan/Desktop/PolyBench/PolyBench-ACC-master/common --cuda-gpu-arch=sm_86
```

##converter.sh

This script automates the processing of CUDA source files using the CUDA Integrated Transformer Tool (`CUDAIntegratedTransformerTool`). It applies various transformations to the source files with different combinations of flags and saves the output in the specified output directory.
## Prerequisites

- [CUDA Integrated Transformer Tool]

## Usage
```bash
./converter.sh INPUT_DIR OUTPUT_DIR INCLUDE_DIR
```

`INPUT_DIR`: Directory containing the CUDA source files to be processed.
`OUTPUT_DIR`: Directory to save the processed CUDA files.
`INCLUDE_DIR`: Directory containing additional header files needed for compilation.

#Example Usage

```bash
./converter.sh /home/furkan/Desktop/PolyBench/PolyBench-ACC-master/CUDA/datamining/covariance /home/furkan/Desktop/Output /home/furkan/Desktop/PolyBench/PolyBench-ACC-master/common
```


# Contributors

We would like to thank the following individuals for their contributions to this project:

- Dr.Isil OZ: Coordinator of the Tool Project.
- Furkan TEMEL: Research and Development Intern
- Gokay GULSOY: Research and Development Intern



