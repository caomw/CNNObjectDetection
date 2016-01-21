#pragma once
#include "Common.h"
#include <assert.h>

void ClassifySoftmaxRegression(CMatLoader &feature, CMatLoader &theta, vector <short> &result, vector <float> &error);
void ClassifySoftmaxRegression(CMatLoader &featureLarge, int rowStart, int colStart, int poolRowSize, int poolColSize, CMatLoader &theta, vector <short> &result, vector <float> &error);
void SerializeSTLImage(CMatLoader &images, vector <BYTE *> &vecImages);
void ClassifyMeans(CMatLoader &featureLarge, int rowStart, int colStart, int poolRowSize, int poolColSize, CMatLoader &meanFeature, vector <short> &result);
void CNNConvolution(BYTE *rgb, int width, int height, CMatLoader &convolvedFeatures);
void ClassifySoftmaxRegressionSingle(CMatLoader &feature, CMatLoader &theta, vector <short> &result, vector <float> &confidence);


class CIntegralFeature
{
public:
	double *integralFeature;
	int width, height;
	CIntegralFeature() {integralFeature = NULL;};
	void Create(int width1, int height1) {
		width = width1;
		height = height1;
		integralFeature = new double[width*height];
	};

	~CIntegralFeature() {
		SAFE_DELETE(integralFeature);
	};

	double GetBlockMeanByDirect(const double *srcImage, const int width, const int height, const int x, const int y, const int w, const int h);

	void CalculateIntegralImage(const double *srcImage);


	inline double GetBlockMeanByIntegralImage(const int x, const int y, const int w, const int h)
	{
		// check exception in debugging time (we assume x, y > 0 for speed-up) 
		// assert( y + h - 1 >= 0 && x + w - 1 >= 0);
		double ret;
		if (y == 0 && x == 0)
			ret = integralFeature[(y + h - 1) * width + (x + w - 1)];
		else if (y == 0)
			ret = integralFeature[(y + h - 1) * width + (x + w - 1)] - integralFeature[(y + h - 1) * width + (x - 1)];
		else if (x == 0)
			ret = integralFeature[(y + h - 1) * width + (x + w - 1)] - integralFeature[(y - 1) * width + (x + w - 1)];
		else
			ret = integralFeature[(y + h - 1) * width + (x + w - 1)] - integralFeature[(y - 1) * width + (x + w - 1)] - integralFeature[(y + h - 1) * width + (x - 1)] + integralFeature[(y - 1) * width + (x - 1)];

		return (ret / w / h);
	};

};