// ImageLoader.h: interface for the CImageLoader class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_IMAGELOADER_H__9E921EB6_CDFF_475F_90BB_B0DE4398B669__INCLUDED_)
#define AFX_IMAGELOADER_H__9E921EB6_CDFF_475F_90BB_B0DE4398B669__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "StdAfx.h"
// #include "ImageTestAlgorithm.h"

typedef enum _BAYER_PIXEL_ORDER
{
	BAYER_ORDER_RGGB = 1,
	BAYER_ORDER_BGGR = 2,
	BAYER_ORDER_GRBG
} BAYER_PIXEL_ORDER;


enum BAYER_TYPE {
	BAYER_TYPE_R = 0,
		BAYER_TYPE_Gr,
		BAYER_TYPE_Gb,
		BAYER_TYPE_B
};



typedef enum _BAYER_PIXEL_BITS
{
	BAYER_PIXEL_BITS_8 = 1,
	BAYER_PIXEL_BITS_10
} BAYER_PIXEL_BIT_COUNT;

typedef enum _IMAGE_TYPE
{
	IMAGE_TYPE_BMP = 0,
	IMAGE_TYPE_RAW
} IMAGE_TYPE;


class CImageObject  
{
private:
	//int ImageType;
public:
	int m_ImageWidth, m_ImageHeight;
	

	int m_ColorChannel;	// gray 1 채널, rgb 3채널 	

	// R, G, B 3 바이트 image pointer
	BYTE *m_rgb_pixel;

	// 3바이트 WidthBytes  맞추어진 display용 image pointer (그냥 display 용)
	BYTE *m_display_pixel;

	

	
	void UpdateDisplayPixel();
	

public:
	IMAGE_TYPE ImageType;


	CImageObject();
	virtual ~CImageObject();

	void InitializeRawImage( int width, int height, BAYER_PIXEL_BIT_COUNT bit_size, BAYER_PIXEL_ORDER bit_order);
	
	void DrawImage(HDC hdc, int width, int height);
	
	bool LoadImageBMP( char *filename );
	BYTE *GetBMPPixelPtr() {return m_rgb_pixel;};
	

	//WORD *GetRAWPixelPtr10Bit() {return m_bayer_pixel_10bit;};	
	
	inline int Width() {return m_ImageWidth;};
	inline int Height() {return m_ImageHeight;};
	void ConvertToGrayImage();
	
	void SaveToBMP(char *filename);
	void SaveToRAW(char *filename);
	bool LoadImage( char *filename);

	void Duplicate(CImageObject &obj);
	void LoadBMPImageFromBuffer(BYTE *pPixel, int width, int height);
	bool CreateReshape( BYTE *pixel, int width, int height, int pixelByteCount );
	

};

#endif // !defined(AFX_IMAGELOADER_H__9E921EB6_CDFF_475F_90BB_B0DE4398B669__INCLUDED_)


