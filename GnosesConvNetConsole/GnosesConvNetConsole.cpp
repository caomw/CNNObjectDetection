// GnosesConvNetConsole.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ImageObject.h"
#include "CNNClassification.h"
#include <direct.h>
#include <algorithm>

void ClassifySTL();
void ClassifyIntegralFeature(char *id);

char className[6][20] = {"", "Airplane", "Car", "Cat", "Dog", "None"};



int _tmain(int argc, _TCHAR* argv[])
{
	printf(" [ Gnoses ConvNet ]   for Daum Interview\n");
	//if (!m_Image.LoadImage("temp.bmp")) return -1;
	printf("Initializing CUDA engine... ");
	InitializeCublas();
	printf("OK\n");

	int operation = 2;
	

	switch (operation)
	{
	case 1:
		ClassifySTL();
		break;
	case 2:
		{
			int i=2;
			char filename[100];

			//for (i=1;i<=1;i++)
			{
				sprintf(filename, "%d.bmp",i);
				ClassifyIntegralFeature(filename);
			}
		}
		break;
	}


	//Classification(pooledFeaturesTrain, softmaxOptTheta);
	//Classification(pooledFeaturesTest, softmaxOptTheta);
	//w.ShowCheckTime("Total time");

	ShutdownCublas();

	printf("Operation completed\n");
	getchar();
	return 0;
}

// ------------------------------------------------------------------------------------------
// Classify STL Test Image Set and save each file separated class into Image/class no. folder
// 
void ClassifySTL()
{
	CImageObject debugImage;
	//debugImage.CreateReshape(serialized, rt.Width(), rt.Height(), 3);	
	//m_ImageDebugDlg.DrawImage(debugImage);

#if 1
	
	CMatLoader inputImages("trainImages2000");
	CMatLoader inputLabels("trainLabels2000");
	CMatLoader pooledFeatures("pooledFeaturesTrain2000");
	CMatLoader softmaxOptTheta("softmaxOptTheta2000");
#else
	CMatLoader inputImages("testImages");
	CMatLoader inputLabels("testLabels");
	// 3600 x m	
	CMatLoader pooledFeatures("pooledFeaturesTest");	
	CMatLoader softmaxOptTheta("softmaxOptTheta2000");
#endif

	// 4 x 3600
	

	vector <BYTE *> vecImages;
	SerializeSTLImage(inputImages, vecImages);

	// CMatLoader testImages("testImages");

	CStopWatch w;
	vector <short> vecLabel;
	vector <float> vecConfidence;
	
	ClassifySoftmaxRegression(pooledFeatures, softmaxOptTheta, vecLabel, vecConfidence);

	int count = 0;
	int i;
	//CLog log("result2.csv", true);
	for (i=0;i<(int)vecLabel.size();i++)
	{
		if (vecLabel[i] == (int)inputLabels.data[i])
			count++;
		//log.WriteLog("%d, %d, %d\n", vecLabel[i] , (int)inputLabels.data[i], vecLabel[i] - (int)inputLabels.data[i]);
	}

	
	printf("Classification finished     accuracy : %f %%\n", (float)count / (float)vecLabel.size() * 100.0f);
	printf("Elapsed time %.0f msec\n", w.CheckTime());

	mkdir("ResultImage");
	mkdir("ResultImage\\1");
	mkdir("ResultImage\\2");
	mkdir("ResultImage\\3");
	mkdir("ResultImage\\4");
	mkdir("ResultImage\\5");
	if (1)
	{
		for (i=0;i<(int)vecImages.size();i++)
			//for (i=0;i<100;i++)
		{
			debugImage.CreateReshape(vecImages[i], 64, 64, 3);
			CString str;
			str.Format("ResultImage\\%d\\%d.bmp", vecLabel[i],i);
			debugImage.SaveToBMP(str.GetBuffer(0));
			
		}
		// Classification(pooledFeaturesTest, softmaxOptTheta, vecLabel);
	}
	printf("%d classified images are saved into ResultImage/classNo directory\n",(int)vecImages.size());

	for (i = 0;i<(int)vecImages.size();i++)
		delete [] vecImages[i];

	return;

} 

void ClassifyIntegralFeature(char *id)
{
	float thr =  0.99f;



	//LPCTSTR file = __targv[1];
	char filename[100];
	CString str(id);
	str = str.Left(str.Find("."));
	sprintf(filename, "InputImage\\%s.bmp",str.GetBuffer(0));
	CImageObject m_Image;
	m_Image.LoadImage(filename);
	printf("Load input image : %s\n", filename);

	//SerializeImage(m_Image.GetBMPPixelPtr(), m_Image.Width(), m_Image.Height(), rt, serialized);
	//m_Image.CreateReshape(serialized, rt.Width(), rt.Height(), 3);	
	CMatLoader convolvedFeatures;
	
	//CNNConvolution(m_Image.m_rgb_pixel,m_Image.Width(), m_Image.Height(),convolvedFeatures);

	//SerializeSTLImage(trainImages, vecImages);
	//    featureLength x imageNum x pooledCountX x pooledCountY    (400 x 2000 x 3 x 3)
	// -> featureLength x pooledCountX x pooledCountY x imageNum    (400 x 3 x 3 x 2000)
	// -> [featureLength x pooledCountX x pooledCountY] x imageNum    
	CMatLoader convolvedFeaturesMatlab(str.GetBuffer(0),"InputImage");
	CMatLoader meanFeature;
	int featureSize = convolvedFeaturesMatlab.dim[0];
	meanFeature.Create(featureSize * 3 * 3,1);

	//CMatLoader pooledFeaturesCatLarge(filename);
	int height = convolvedFeaturesMatlab.dim[2];
	int width = convolvedFeaturesMatlab.dim[3];
	//pooledFeaturesCatLarge.Seriallize();
	int row, col;

	CMatLoader softmaxOptTheta("softmaxOptThetaNoise");
	CMatLoader softmaxOptTheta5320("softmaxOptTheta5320");
	

	vector <short> vecLabel;
	vector <float> vecConfidence;

	CStopWatch w;

	
	
	CIntegralFeature *features = new CIntegralFeature [featureSize];
	
	int i;
	

	// Calculate integral image in separated features
	for (i = 0;i< featureSize;i++)	
	{
		double *separatedFeature = new double[height * width];
		features[i].Create(width, height);
			
		for (col = 0;col < width;col++)
		{
			
			for (row = 0;row < height;row++)
			{
				// row -> col order
				separatedFeature[row * width + col] = convolvedFeaturesMatlab.data[(col * height + row) * featureSize + i];
				//if (i == 13)
				//log.WriteLog("%d,%d,%d,%f\n", i, row, col,convolvedFeatures.data[(col * height + row) * featureSize + i]);
				
			}
		}
		features[i].CalculateIntegralImage(separatedFeature);

		delete [] separatedFeature;
	}


	CLog log("confidence.csv", 1);

	// Calculate mean feature in box area with k x k size
	int k;
	//k = 128/3 - 5;//min(width,height) /2;
	// int step = 19;
	int start = min(width,height) / 3 - 5;
	int end = start / 2;
	int candidate = 0;
	vector <int> vecCandidateLabel;
	
	BOOL found = FALSE;
	CRectEx rtCadidate;
	//vector <CRectEx> vecRect;
	//vector <double> vecScore[4];
	CRectEx rtFinal;

	for (k = start;k>= end ;k-=10)	
	{
		for (row = 0;row <= height - (k * 3) ;row+=k/2)
		{
			for (col = 0;col <= width - (k * 3);col+=k/2)
			{
				int row2, col2;
				int featureIndex = 0;
				for (col2 = 0;col2 < 3;col2++)
				{
					
					for (row2 = 0;row2 < 3;row2++)
					{
						//log.WriteLog("%d,%d,", row,col);		
						//log.WriteLog("%d,%d,", col + (col2 * k), row + (row2 * k));
						for (i=0;i<featureSize;i++)
						{
							meanFeature.data[featureIndex++] = (features[i].GetBlockMeanByIntegralImage(col + (col2 * k), row + (row2 * k), k, k));
							//if (col == 95 && row == 0)
							//	log.WriteLog("%f\n", meanFeature.data[i]);
						}
					}
					
				}
				//log.WriteLog("\n");
				//ClassifySoftmaxRegressionSingle(meanFeature, softmaxOptTheta, vecLabel, vecConfidence);
				//if (vecLabel[0] == 5 ) continue;
				CRectEx rt;
				rt.SetRect(0,0,k*3+8,k*3+8);
				rt.Offset(col,row);

				//m_OvrDisp.DrawRect(rt,color[vecLabel[0]], 10,"%s(%d)",className[vecLabel[0]], vecLabel[0]);
				//log.WriteLog("%s,%d, %.1f, %.1f, %.1f, %.1f,%.1f,",className[vecLabel[0]], vecLabel[0], vecConfidence[0],vecConfidence[1],vecConfidence[2],vecConfidence[3], vecConfidence[4]);

				ClassifySoftmaxRegressionSingle(meanFeature, softmaxOptTheta5320, vecLabel, vecConfidence);
				if (k==start)
					vecCandidateLabel.push_back(vecLabel[0]-1);
				else if (vecLabel[0] == candidate)
				{
					//m_OvrDisp.DrawRect(rt,color[vecLabel[0]], 10,"%s(%d) %.1f %.1f %.1f %.1f",className[vecLabel[0]], vecLabel[0], vecConfidence[0]);
					//vecRect.push_back(rt);
					rtFinal = rt;
					found = TRUE;
				}

				log.WriteLog(",%s,%d,%.1f, %.1f, %.1f, %.1f\n",className[vecLabel[0]], vecLabel[0], vecConfidence[0],vecConfidence[1],vecConfidence[2],vecConfidence[3]);
				if (vecLabel[0] != 5 && vecConfidence[0] > 0.99)		
				{					
					//ClassifySoftmaxRegression(meanFeature, softmaxOptTheta100percentTraining, vecLabel, vecConfidence);
				//	m_OvrDisp.DrawRect(rt,color[vecLabel[0]], 10,"%s(%d) %.1f %.1f %.1f %.1f",className[vecLabel[0]], vecLabel[0], vecConfidence[0]);
				//	goto FINISH;
				}
				else
				{
					//m_OvrDisp.DrawRect(rt,color[vecLabel[0]], 10,"%s(%d) %f",className[vecLabel[0]], vecLabel[0], vecConfidence[0]);
				}


				//m_OvrDisp.DrawRect(rt,MCYAN);
				//log.WriteLog("%d,%f\n",vecLabel[0],vecConfidence[0]);
				
			}
		}

		if (k == start)
		{
			vector <int> histo;
			histo.resize(4);
			for (int l=0;l<(int)vecCandidateLabel.size();l++)
				histo[vecCandidateLabel[l]]++;

			candidate = distance(histo.begin(), max_element(histo.begin(), histo.end())) + 1;
			rtCadidate.SetRect(0,0,k*3+8,k*3+8);
			//rtCadidate.Offset(col,row);
		}
		
	}

	if (found == FALSE)
		printf("%d, %d, %d %d = %s\n", rtCadidate.left,rtCadidate.top,rtCadidate.right,rtCadidate.bottom, className[candidate]);
	else
		printf("%d, %d, %d %d = %s\n", rtCadidate.left,rtFinal.top,rtFinal.right,rtFinal.bottom, className[candidate]);

	printf("Elapsed time in %.0f msec (%d x %d image)\n", w.CheckTime(),m_Image.Width(), m_Image.Height());

	/*for (row = 0;row <= rowSize-3;row+=step)
		//for (row = 0;row < 1;row++)
	{
		for (col = 0;col <= colSize-3;col+=step)
			//for (col = 0;col < 1;col++)
		{
			//w.StartTime();
			//double error = ClassifyMeans(pooledFeaturesLarge,row, col,3,3, meanFeature, vecLabel);
			//if (vecLabel[0] == 0 ) continue;
			ClassifySoftmaxRegression(convolvedFeatures,row, col,3,3, softmaxOptTheta, vecLabel, vecConfidence);
			//if (vecLabel[0] == 5) continue;
			//if (vecConfidence[0] < thr) continue;

			//m_OvrDisp.DrawText(10 + (row *10),10, MGREEN, 20, "%.0f msec", w.CheckTime());

			CRectEx rt;
			rt.SetRect(0,0,63,63);
			rt.Offset(int((float)col / 3.0f * 64.0f),int((float)row / 3.0f * 64.0f));

			m_OvrDisp.DrawRect(rt,color[vecLabel[0]], 10,"%s(%d) %f",className[vecLabel[0]], vecLabel[0], vecConfidence[0]);
			//m_OvrDisp.DrawText(rt.CenterPoint().x,rt.CenterPoint().y,MGREEN, 10,"%s",className[vecLabel[0]]);
			//m_OvrDisp.DrawRect(rt,MRED, 10,"%d,%d,%d,%d",vecLabel[0], vecLabel[1],vecLabel[2],vecLabel[3]);
			//m_OvrDisp.DrawRect(rt,MRED);
		}
	}*/

	//m_OvrDisp.DrawText(10,10, MCYAN, 15, "%d x %d %.0f msec", m_Image.Width(), m_Image.Height(),w.CheckTime());
	delete [] features;
	return;

} 
