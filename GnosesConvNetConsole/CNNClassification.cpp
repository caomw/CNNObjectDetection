#include "stdafx.h"
#include "cuBlasWrapper.h"
#include <algorithm>
#include "CNNClassification.h"


void SerializeSTLImage(CMatLoader &images, vector <BYTE *> &vecImages)
{
	int i,j;
	double *src = images.data;
	int imageSize = 64 * 64 * 3;
	vecImages.resize(images.dim[3]);
	for (i=0;i<images.dim[3];i++)
	{
		vecImages[i] = new BYTE[imageSize];
		for (j=0;j<imageSize;j++)
		{
			vecImages[i][j] = BYTE(*src * 255.0);
			src ++;
		}		

		//src += imageSize;
	}
}


// classify stl images (image matrix)
void ClassifySoftmaxRegression(CMatLoader &feature, CMatLoader &theta, vector <short> &result, vector <float> &confidence)
{
	CCudaMatrix matFeature;
	CCudaMatrix matTheta;

	matFeature.Create(feature.dim[0], feature.dim[1], feature.data);
	matTheta.Create(theta.dim[0], theta.dim[1], theta.data);

	CCudaMatrix matPredict;


	MatrixProduct(matTheta, matFeature, matPredict);
	

	float *memc;
	float *memd;
	matPredict.ExportData(&memc);
	matPredict.ExportData(&memd);

	// compare with dirct calculation
// 	float *mema;
// 	matTheta.ExportData(&mema);
// 
// 
// 	float *memb;
// 	matFeature.ExportData(&memb);
// 	SimpleMatrixMultiply(mema, memb, memd, matTheta.m,matTheta.n,matFeature.n);
	// ------------------------------
	
	

	double sum = 0;
	float *ptr;
	for (int i=0;i<matPredict.n;i++)
	{
		ptr = &memc[i * matPredict.m];
		sum = 0;
		double maxValue = *max_element(ptr, ptr + matPredict.m);
		for (int j=0;j<matPredict.m;j++)
		{
			*(ptr + j) -= (float)maxValue;
			sum += (double)exp((double)*(ptr+j));
			
		}

		for (int j=0;j<matPredict.m;j++)
		{			
			//*(ptr+j) = (float)((double)exp((double)*(ptr+j)) / sum);
			*(ptr+j) = (float)((double)exp((double)*(ptr+j)));
			//log.WriteLog("%f,", memc[i * matPredict.m + j]);
		}

		//log.WriteLog("\n");
	}

	/*CLog log("result.csv");
	for (int i=0;i<matPredict.n;i++)
	{
		for (int j=0;j<matPredict.m;j++)
			log.WriteLog("%f,", memc[i * matPredict.m + j]);
		log.WriteLog("\n");
	}*/


	result.resize(matPredict.n);
	confidence.resize(matPredict.n);
	if (1)
	{
		
		for (int i=0;i<matPredict.n;i++)
		{
			float *data = &memc[i * matPredict.m];

			//cout << distance(data, max_element(data, data+5)) + 1 << ",";
			float *maxerror = max_element(data, data + matPredict.m);
			confidence[i] = *maxerror;
			result[i] = distance(data, maxerror) + 1;
			
		}
	}
	free(memc);
	
}

// classify single image
void ClassifySoftmaxRegressionSingle(CMatLoader &feature, CMatLoader &theta, vector <short> &result, vector <float> &confidence)
{
	CCudaMatrix matFeature;
	CCudaMatrix matTheta;

	matFeature.Create(feature.dim[0], feature.dim[1], feature.data);
	matTheta.Create(theta.dim[0], theta.dim[1], theta.data);

	CCudaMatrix matPredict;


	MatrixProduct(matTheta, matFeature, matPredict);
	

	float *memc;
	float *memd;
	matPredict.ExportData(&memc);
	matPredict.ExportData(&memd);

	// compare with dirct calculation
// 	float *mema;
// 	matTheta.ExportData(&mema);
// 
// 
// 	float *memb;
// 	matFeature.ExportData(&memb);
// 	SimpleMatrixMultiply(mema, memb, memd, matTheta.m,matTheta.n,matFeature.n);
	// ------------------------------
	
	

	double sum = 0;
	result.resize(matPredict.n);
	confidence.resize(matPredict.m);
	
	for (int j=0;j<matPredict.m;j++)
	{
		confidence[j] = memc[j];

	}

	//for (int i=0;i<matPredict.n;i++)
	{
		float *maxValue = max_element(memc, memc + matPredict.m);
		result[0] = distance(memc, maxValue) + 1;

		/*for (int j=0;j<matPredict.m;j++)
		{
			*(ptr + j) -= (float)maxValue;
			sum += (double)exp((double)*(ptr+j));
			
		}

		for (int j=0;j<matPredict.m;j++)
		{			
			//*(ptr+j) = (float)((double)exp((double)*(ptr+j)) / sum);
			*(ptr+j) = (float)((double)exp((double)*(ptr+j)));
			//log.WriteLog("%f,", memc[i * matPredict.m + j]);
		}*/

		//log.WriteLog("\n");
	}

	/*CLog log("result.csv");
	for (int i=0;i<matPredict.n;i++)
	{
		for (int j=0;j<matPredict.m;j++)
			log.WriteLog("%f,", memc[i * matPredict.m + j]);
		log.WriteLog("\n");
	}*/


	free(memc);
	
}

void ClassifyMeans(CMatLoader &featureLarge, int rowStart, int colStart, int poolRowSize, int poolColSize, CMatLoader &meanFeature, vector <short> &result)
{
	int featureLength = featureLarge.dim[0];
	int rowSize = featureLarge.dim[1];
	int colSize = featureLarge.dim[2];

	//featureLarge.Seriallize();

	CMatLoader feature;

	feature.Create(featureLength*3*3,1);
	result.resize(4);

	int row2, col2, i, j,k;
	double error = 0, error2 = 0;
	for (row2 = 0;row2 < poolRowSize;row2++)
	{

		for (col2 = 0;col2 < poolColSize;col2++)
		{
			i = ((colStart + col2) * rowSize + (rowStart + row2)) * featureLength;
			memcpy(&feature.data[(col2*3+row2)*featureLength], featureLarge.data + i, sizeof(double) * featureLength);
			
			
		}
	}

	for (k=0;k<4;k++)
	{
		error2 = 0;
		for (j=0;j<meanFeature.dim[0];j++)
		{
			double value = meanFeature.data[j] - featureLarge.data[k * meanFeature.dim[0] + (i + j)];
			error2 += (value * value);
		}
		error = sqrt(error2);
		result[k] = (int)(error * 100.0);
	}
	
	//result[0] = 1;
	
}


void ClassifySoftmaxRegression(CMatLoader &featureLarge, int rowStart, int colStart, int poolRowSize, int poolColSize, CMatLoader &theta, vector <short> &result, vector <float> &confidence)
{
	int featureLength = featureLarge.dim[0];
	int rowSize = featureLarge.dim[1];
	int colSize = featureLarge.dim[2];
	
	//featureLarge.Seriallize();
	
	CMatLoader feature;
	
	feature.Create(featureLength*3*3,1);

	int row2, col2, i;

	for (row2 = 0;row2 < poolRowSize;row2++)
	{

		for (col2 = 0;col2 < poolColSize;col2++)
		{
			i = ((colStart + col2) * rowSize + (rowStart + row2)) * featureLength;
			memcpy(&feature.data[(col2*3+row2)*featureLength], featureLarge.data + i, sizeof(double) * featureLength);
		}
	}
	
	ClassifySoftmaxRegression(feature, theta, result, confidence);
	
}

void Convolution(double *pixel, int width, int height, double *kernel, int kernelSize, double **convolvedAddr)
{
	int row, col;
	int u, v;
	double *convolved = *convolvedAddr;
	int convolvedWidth = width - kernelSize + 1;
	int convolvedHeight = height - kernelSize + 1;

	for (row = 0;row<convolvedHeight;row++)
	{
		for (col= 0;col<=convolvedWidth;col++)
		{
			for (v = 0;v<kernelSize;v++)
			{
				for (u = 0;u<kernelSize;u++)
				{
					convolved[row*width+col] = double(pixel[(row + v) * width + (col + u)] * kernel[v * kernelSize + u]);
				}
			}
		}
	}
}

// CNN Convolutional feature creation
void CNNConvolution(BYTE *rgb, int width, int height, CMatLoader &convolvedFeatures)
{
	CMatLoader precomputed("precomputed");
	CMatLoader WT("WT");
	CMatLoader SAEFeature("SAEFeature");
	int kernelSize = 8;
	int convolvedSizeX = width - kernelSize + 1;
	int convolvedSizeY = height - kernelSize + 1;
	int featureSize = WT.dim[0];
	CMatLoader singleFeature;
	singleFeature.Create(convolvedSizeX, convolvedSizeY);

	// channel floating image
	CMatLoader image[3];
	CMatLoader convolvedch[3];
	int i,f, ch;
	int row, col;
	
	
	for (ch=0;ch<3;ch++)
	{
		image[ch].Create(height, width);
		convolvedch[ch].Create(convolvedSizeX, convolvedSizeY);

		for (i=0;i<width*height;i++)
		{
			image[ch].data[i] = (double)rgb[i * 3 + ch] / 255.0;
		}
	}

	convolvedFeatures.Create(featureSize, convolvedSizeX * convolvedSizeY);

	for (f=0;f<featureSize;f++)
	{
		//ZeroMemory(convolved.data,sizeof(double) * convolved.dim[0];
		for (ch=0;ch<3;ch++)
		{
			Convolution(image[ch].data, width, height, WT.data + (f * 192 + ch * 64),8,&convolvedch[ch].data);
			for (row = 0;row<convolvedSizeY;row++)
			{
				for (col= 0;col<=convolvedSizeX;col++)
				{
					convolvedFeatures.data[featureSize * (col * convolvedSizeY + row) + f] += convolvedch[ch].data[row * convolvedSizeX + col];
				}
			}
		}
		
	}
	return;
}


//void mainCNN()
void main2()
{
	InitializeCublas();

	CStopWatch w;
	CMatLoader pooledFeaturesTrain("pooledFeaturesTrain");
	CMatLoader pooledFeaturesTest("pooledFeaturesTest");	
	CMatLoader softmaxOptTheta("softmaxOptTheta");
	
	//Classification(pooledFeaturesTrain, softmaxOptTheta);
	//Classification(pooledFeaturesTest, softmaxOptTheta);
	//w.ShowCheckTime("Total time");
	ShutdownCublas();

}

double CIntegralFeature::GetBlockMeanByDirect( const double *srcImage, const int width, const int height, const int x, const int y, const int w, const int h )
{
	// check exception in debugging time (we assume x, y > 0 for speed-up) 
	int row, col;
	int sum = 0;
	for (row = y ; row < y + h ; row++)
	{		
		for (col = x ; col < x + w ; col++)
		{
			sum += (int)srcImage[row * width + col];
		}
	}
	return sum;
}

void CIntegralFeature::CalculateIntegralImage( const double *srcImage )
{
	int row, col;
	double *line = new double[width];		
	integralFeature[0] = srcImage[0];
	double direct = 0;
	for (col = 1 ; col < width ; col++)
	{
		integralFeature[col] = integralFeature[col-1] + srcImage[col];
		//direct =  GetBlockMeanByDirect(srcImage, width, height, 0, 0, col + 1, 1);
	}

	for (row = 1 ; row < height ; row++)
	{
		line[0] = srcImage[row*width];
		integralFeature[row*width] = integralFeature[(row-1)*width] + srcImage[row*width];
		direct =  GetBlockMeanByDirect(srcImage, width, height, 0, 0, col + 1, row + 1);
		for (col = 1 ; col < width ; col++)
		{
			line[col] = line[col-1] + srcImage[row*width+col];

			integralFeature[row*width+col] = integralFeature[(row-1)*width+col] + line[col-1] + srcImage[row*width+col];
			//direct =  GetBlockMeanByDirect(srcImage, width, height, 0, 0, col + 1, row + 1);
			//assert((double)integralFeature[row*width+col] != direct);

		}
	}
	delete [] line;

	//cout << "Integral image initialize completed." << endl;
}
