/* Includes, system */
#pragma  once
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"





class CCudaMatrix
{


public:
	float *data;
	int m, n;
	CCudaMatrix() {data = NULL;}; 
	~CCudaMatrix() { if (data != NULL) cudaFree(data); };
	bool Create(int m1, int n1);;
	bool Create( int m1, int n1, double *data );
	void Rand();;

	void ExportData(float **mem);
	void PrintMatrix(const float *A, int nr_rows_A, int nr_cols_A);
	float MatirxNorm();
	void ImportData( double *mem );
	double Mean();
};

void MatrixProduct(CCudaMatrix & a, CCudaMatrix & b, CCudaMatrix & c);
float MatirxNorm(CCudaMatrix & a);
void InitializeCublas();
void ShutdownCublas();
float SimpleNorm(const float *a, const int m, const int n);
void SimpleMatrixMultiply(const float *A, const float *B, float *C, const int m, const int k, const int n);

void InitializeCublas();
void ShutdownCublas();

