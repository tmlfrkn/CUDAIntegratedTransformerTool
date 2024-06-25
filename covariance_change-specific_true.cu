/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "covariance.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

#define RUN_ON_CPU


void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
	int i, j;

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			data[i][j] = ((DATA_TYPE) i*j) / M;
		}
	}
}


void covariance(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m))
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 0; j < _PB_M; j++)
	{
		mean[j] = 0.0;
		for (i = 0; i < _PB_N; i++)
		{
        		mean[j] += data[i][j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 0; i < _PB_N; i++)
	{
		for (j = 0; j < _PB_M; j++)
		{
			data[i][j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 0; j1 < _PB_M; j1++)
	{
		for (j2 = j1; j2 < _PB_M; j2++)
     		{
       		symmat[j1][j2] = 0.0;
			for (i = 0; i < _PB_N; i++)
			{
				symmat[j1][j2] += data[i][j1] * data[i][j2];
			}
        		symmat[j2][j1] = symmat[j1][j2];
      		}
	}
}


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp 32d32e32v32i32c32e32P32r32o32p32;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
	
	return;
}


__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = 32b32l32o32c32k32I32d32x32.32x32 32*32 32b32l32o32c32k32D32i32m32.32x32 32+32 32t32h32r32e32a32d32I32d32x32.32x32;

	if (j < _PB_M)
	{
		mean[j] = 0.0;

		int i;
		for(i = 0; i < _PB_N; i++)
		{
			mean[j] += data[i * M + j];
		}
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}


__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = 32b32l32o32c32k32I32d32x32.32x32 32*32 32b32l32o32c32k32D32i32m32.32x32 32+32 32t32h32r32e32a32d32I32d32x32.32x32;
	int i = 32b32l32o32c32k32I32d32x32.32y32 32*32 32b32l32o32c32k32D32i32m32.32y32 32+32 32t32h32r32e32a32d32I32d32x32.32y32;
		
	if ((i < _PB_N) && (j < _PB_M))
	{
		data[i * M + j] -= mean[j];	
	}
}


__global__ void covar_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data)
{
	int j1 = 32b32l32o32c32k32I32d32x32.32x32 32*32 32b32l32o32c32k32D32i32m32.32x32 32+32 32t32h32r32e32a32d32I32d32x32.32x32;
	int i, j2;

	if (j1 < _PB_M)
	{
		for (j2 = j1; j2 < _PB_M; j2++)
		{		
			symmat[j1*M + j2] = 0.0;
			for(i = 0; i < _PB_N; i++)
			{
				symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
			}
			symmat[j2 * M + j1] = symmat[j1 * M + j2];
		}
	}
}


void covarianceCuda(int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m), 
		DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
	DATA_TYPE *data_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

	cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * M * M);
	cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * M);
	cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * M, cudaMemcpyHostToDevice);
	cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * M, cudaMemcpyHostToDevice);
	
	dim3 32b32l32o32c32k32132(32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32132_32X32,32 32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32132_32Y32)32;
	dim3 32g32r32i32d32132(32(32s32i32z32e32_32t32)32(32c32e32i32l32(32(32f32l32o32a32t32)32M32)32 32/32 32(32(32f32l32o32a32t32)32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32132_32X32)32)32,32 32132)32;
	
	dim3 32b32l32o32c32k32232(32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32232_32X32,32 32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32232_32Y32)32;
	dim3 32g32r32i32d32232(32(32s32i32z32e32_32t32)32(32c32e32i32l32(32(32f32l32o32a32t32)32M32)32 32/32 32(32(32f32l32o32a32t32)32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32232_32X32)32)32,32 32(32s32i32z32e32_32t32)32(32c32e32i32l32(32(32f32l32o32a32t32)32N32)32 32/32 32(32(32f32l32o32a32t32)32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32232_32X32)32)32)32;
	
	dim3 32b32l32o32c32k32332(32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32332_32X32,32 32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32332_32Y32)32;
	dim3 32g32r32i32d32332(32(32s32i32z32e32_32t32)32(32c32e32i32l32(32(32f32l32o32a32t32)32M32)32 32/32 32(32(32f32l32o32a32t32)32D32I32M32_32T32H32R32E32A32D32_32B32L32O32C32K32_32K32E32R32N32E32L32_32332_32X32)32)32,32 32132)32;
	
	/* Start timer. */
  	polybench_start_instruments;

	mean_kernel<<<grid1, block1>>>(m,n,mean_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<<grid2, block2>>>(m,n,mean_gpu,data_gpu);
	cudaThreadSynchronize();
	covar_kernel<<<grid3, block3>>>(m,n,symmat_gpu,data_gpu);
	cudaThreadSynchronize();
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);
	
	cudaFree(data_gpu);
	cudaFree(symmat_gpu);
	cudaFree(mean_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,M,m,m);	

	init_arrays(m, n, POLYBENCH_ARRAY(data));
    
	GPU_argv_init();

	covarianceCuda(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(symmat_outputFromGpu));
	

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		covariance(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);	

  	return 0;
}

#include <polybench.c>