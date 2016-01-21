#include "stdafx.h"
#include "cuBlasWrapper.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <iostream>
#include <curand.h>
#include <time.h>
#include <algorithm>
#include <numeric>
// curand.lib;cublas.lib;cuda.lib;cudart.lib
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")

using namespace std;

cublasHandle_t handle = NULL;


void InitializeCublas()
{
	cublasCreate(&handle);
}

void ShutdownCublas()
{
	// Destroy the handle
	cublasDestroy(handle);
}

inline void CublasMarixMultiply(const float *A, const float *B, float *C, const int m, const int k, const int n) 
{

	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	
	
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);



}



void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}


void MatrixProduct(CCudaMatrix & a, CCudaMatrix & b, CCudaMatrix & c)
{
	if (c.data == NULL)
		c.Create(a.m, b.n);
	CublasMarixMultiply(a.data, b.data, c.data, a.m, a.n, b.n);

};

float CCudaMatrix::MatirxNorm()
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	float result;
	cublasSnrm2(handle, m * n, data, 1, &result);
	// Destroy the handle
	cublasDestroy(handle);
	return result;
};

bool CCudaMatrix::Create( int m1, int n1 )
{
	m = m1; n = n1;
	if (data != NULL) cudaFree(data);

	if (cudaMalloc(&data,m * n * sizeof(float)) != cudaSuccess) 
	{
		printf("cuda alloc failed\n");
		return false;
	}
	Rand();
	return true;
}


bool CCudaMatrix::Create( int m1, int n1, double *src )
{
	m = m1; n = n1;
	if (data != NULL) cudaFree(data);

	if (cudaMalloc(&data,m * n * sizeof(float)) != cudaSuccess) 
	{
		printf("cuda alloc failed\n");
		return false;
	}
	ImportData(src);
	return true;
}

void CCudaMatrix::Rand()
{
	GPU_fill_rand(data, m, n);
}

void CCudaMatrix::ImportData( double *mem )
{
	float *data2 = new float[m*n];
	for (int i=0;i<m*n;i++)
		*(data2 + i) = float(*(mem + i));
	cudaMemcpy(data, data2, m * n * sizeof(float),cudaMemcpyHostToDevice);

	
	delete [] data2;
}


void CCudaMatrix::ExportData( float **mem )
{
	*mem = (float *)malloc(m * n * sizeof(float));		
	cudaMemcpy(*mem,data, m * n* sizeof(float),cudaMemcpyDeviceToHost);
}

double CCudaMatrix::Mean()
{
	float *mem;
	ExportData(&mem);
	return accumulate(mem, mem + (m*n), 0.0);
}





void CCudaMatrix::PrintMatrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for(int i = 0; i < nr_rows_A; ++i){
		for(int j = 0; j < nr_cols_A; ++j){
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}




// [m,k] x [k, n]
void SimpleMatrixMultiply(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
	int row,col, col2;
	float sum = 0;
	//CLog log("matrix.csv",1);
	for (row = 0;row<m;row++)
	{

		for (col2 = 0;col2<n;col2++)		
		{
			sum = 0.0;
			for (col = 0;col<k;col++)
			{
				sum += A[(col*m) + row] * B[(col2*k) + col];
				//log.WriteLog("%f,%f,%f\n", A[(col*m) + row], B[(col2*k) + col], sum);
			}
			C[(col2 * m) + row] = float(sum);
		}		
	}
}


float SimpleNorm(const float *a, const int m, const int n)
{
	int i;
	double sum = 0.0;
	for (i = 0;i<m*n;i++)
	{

		sum += (double)(a[i] * a[i]);

	}

	return (float)(sqrt(sum)/(m*n));
}

