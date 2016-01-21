// ImageLoader.cpp: implementation of the CImageLoader class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "ImageObject.h"


#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif


#define WIDTHBYTES(w,bitcount) ((((w)*(bitcount)+31) & ~31) >> 3)




//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CImageObject::CImageObject()
{
	m_rgb_pixel = NULL;
	m_display_pixel = NULL;

	
}

CImageObject::~CImageObject()
{
	SAFE_DELETE(m_rgb_pixel);
	SAFE_DELETE(m_display_pixel);
	
}



bool CImageObject::LoadImage( char *filename)
{
	CString file(filename);
	CString ext = file.Right(3);
	ImageType = IMAGE_TYPE_BMP;
	ext.MakeLower();
	
	
	if (ext == "bmp")
	{
		return LoadImageBMP(filename);
		
	}
	
	return false;

}



bool CImageObject::LoadImageBMP( char *filename )
{

	BITMAPFILEHEADER bmFileHeader;
	BITMAPINFOHEADER bmInfoHeader;
	
		
	CFile file;
	if (!file.Open(filename, CFile::modeNoTruncate | CFile::modeRead ))  
	{
		printf("No file %s", filename);
		return false;
	}
	file.Read(&bmFileHeader, sizeof(BITMAPFILEHEADER));
	file.Read(&bmInfoHeader, sizeof(BITMAPINFOHEADER));

	m_ImageWidth = bmInfoHeader.biWidth;
	m_ImageHeight = abs(bmInfoHeader.biHeight);
	int bytespixel_src = bmInfoHeader.biBitCount/8;

	BYTE *PixelFlip = new BYTE[m_ImageHeight * m_ImageWidth * bytespixel_src];

	file.Read(PixelFlip, m_ImageHeight * m_ImageWidth * bytespixel_src);
	file.Close();
		
	

	
	
	if (m_rgb_pixel != NULL)	
		delete m_rgb_pixel;

	if (m_display_pixel != NULL)	
		delete m_display_pixel;

	

	int widthbytes_display = WIDTHBYTES(m_ImageWidth,bmInfoHeader.biBitCount);

	m_rgb_pixel = new BYTE[m_ImageHeight * m_ImageWidth * 3];

	// Loading Flip BMP image
	for (int row=0;row<m_ImageHeight;row++)
	{
		for (int col=0;col<m_ImageWidth;col++)
		{
			m_rgb_pixel[3 * ((m_ImageHeight - row - 1)*m_ImageWidth+col)    ]	= PixelFlip[(row*widthbytes_display + col*bytespixel_src) + 0];
			m_rgb_pixel[3 * ((m_ImageHeight - row - 1)*m_ImageWidth+col) + 1] = PixelFlip[(row*widthbytes_display + col*bytespixel_src) + 1];
			m_rgb_pixel[3 * ((m_ImageHeight - row - 1)*m_ImageWidth+col) + 2] = PixelFlip[(row*widthbytes_display + col*bytespixel_src) + 2];

		}
	}
	
	//delete temp2;
	delete [] PixelFlip;
	return true;
	// Invalidate(FALSE);
}




bool CImageObject::CreateReshape( BYTE *pixel, int width, int height, int pixelByteCount )
{
	m_ImageWidth = width;
	m_ImageHeight = height;

	if (m_rgb_pixel != NULL)	
		delete m_rgb_pixel;
	
	if (m_display_pixel != NULL)
		delete m_display_pixel;
	
	m_rgb_pixel = new BYTE[m_ImageHeight * m_ImageWidth * 3];
	
	
	BYTE *ir = pixel;
	BYTE *ig = ir + (width * height);
	BYTE *ib = ig + (width * height);
	// Loading Flip BMP image
	for (int row=0;row<width;row++)
	{
		for (int col=0;col<height;col++)
		{
			m_rgb_pixel[pixelByteCount * (col*m_ImageHeight+row)    ] = *(ir++);
			m_rgb_pixel[pixelByteCount * (col*m_ImageHeight+row) + 1] = *(ig++);
			m_rgb_pixel[pixelByteCount * (col*m_ImageHeight+row) + 2] = *(ib++);
		}
	}
	
	return true;
	// Invalidate(FALSE);
}

// rgb_order : 1 RG GB  ,    2 BG GR
void CImageObject::InitializeRawImage( int width, int height, BAYER_PIXEL_BIT_COUNT bit_size, BAYER_PIXEL_ORDER bit_order)
{
	m_ImageWidth = width;
	m_ImageHeight = height;

	SAFE_DELETE(m_rgb_pixel);
	m_rgb_pixel = new BYTE[m_ImageWidth*m_ImageHeight*3];


	
}
void CImageObject::DrawImage(HDC hdc, int width, int height)
{

	if (m_rgb_pixel == NULL) return;
	int widthbytes = WIDTHBYTES(m_ImageWidth,24);
	
	BITMAPINFO bmi;
	ZeroMemory(&bmi,sizeof(bmi));
	
	bmi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biPlanes=1;
	bmi.bmiHeader.biWidth=m_ImageWidth;
	bmi.bmiHeader.biHeight=-m_ImageHeight;
	bmi.bmiHeader.biBitCount=24;
	bmi.bmiHeader.biCompression=BI_RGB;
	bmi.bmiHeader.biSizeImage=m_ImageHeight*widthbytes;
	SetStretchBltMode(hdc, COLORONCOLOR);
	//SetStretchBltMode(hdc, HALFTONE);

	UpdateDisplayPixel();

	StretchDIBits(
		hdc, // handle of device context
		0,
		0,
		width,
		height,		
		0,
		0,
		m_ImageWidth,
		m_ImageHeight,
		(CONST VOID *)m_display_pixel, // address of array with DIB bits
		&bmi, // address of structure with bitmap info.
		DIB_RGB_COLORS, // RGB or palette indices
		SRCCOPY
	);	
}


void CImageObject::ConvertToGrayImage()
{
	int widthbytes_display = WIDTHBYTES(m_ImageWidth,24);
	m_ColorChannel = 1;
	BYTE *pixel = new BYTE[m_ImageWidth * m_ImageHeight];
	for (int row=0;row<m_ImageHeight;row++)
	{
		for (int col=0;col<m_ImageWidth;col++)
		{
			//float gray = m_rgb_pixel[3 * (row*m_ImageWidth+col) + 1 ];

			float gray = (0.3f  * (float)m_rgb_pixel[3 * (row*m_ImageWidth+col)    ]) + 
				(0.3f  * (float)m_rgb_pixel[3 * (row*m_ImageWidth+col) + 1 ]) + 
				(0.11f * (float)m_rgb_pixel[3 * (row*m_ImageWidth+col) + 2 ]);

			pixel[ row*m_ImageWidth+col ]	= BYTE(ROUND(gray));			
			m_rgb_pixel[(row*m_ImageWidth+col)] = BYTE(gray); 
		}
	}

	delete [] pixel;
}

void CImageObject::UpdateDisplayPixel()
{
	int widthbytes_display = WIDTHBYTES(m_ImageWidth,24);
	SAFE_DELETE(m_display_pixel);
	m_display_pixel = new BYTE[m_ImageHeight * widthbytes_display];
	if (m_ColorChannel == 1)
	{
		for (int row=0;row<m_ImageHeight;row++)
		{
			for (int col=0;col<m_ImageWidth;col++)
			{
				m_display_pixel[row*widthbytes_display + (col*3) + 0] = m_rgb_pixel[ (row*m_ImageWidth+col) ];
				m_display_pixel[row*widthbytes_display + (col*3) + 1] = m_rgb_pixel[ (row*m_ImageWidth+col) ];
				m_display_pixel[row*widthbytes_display + (col*3) + 2] = m_rgb_pixel[ (row*m_ImageWidth+col) ];
			}
		}
	}
	else
	{
		for (int row=0;row<m_ImageHeight;row++)
		{
			for (int col=0;col<m_ImageWidth;col++)
			{
				m_display_pixel[row*widthbytes_display + (col*3) + 0] = m_rgb_pixel[ (row*m_ImageWidth+col) * 3 ];
				m_display_pixel[row*widthbytes_display + (col*3) + 1] = m_rgb_pixel[ (row*m_ImageWidth+col) * 3 + 1 ];
				m_display_pixel[row*widthbytes_display + (col*3) + 2] = m_rgb_pixel[ (row*m_ImageWidth+col) * 3 + 2 ];
			}
		}
	}
}

/*void CImageObject::GetBayerIntensity( CRectEx region, double *R, double *Gr, double *Gb, double *B)
{
	int count[4] = {0,0,0,0};
	double sum[4] = {0,0,0,0};

	for (int row=region.top;row<=region.bottom;row++)
	{
		for (int col=region.left;col<=region.right;col++)
		{			
			int bayer_type = m_bayer_type[row*m_ImageWidth + col];
			
			// Pedestal (Black Level) 16
			m_bayer_pixel[row*m_ImageWidth + col] -= 16;
			if (m_bayer_pixel[row*m_ImageWidth + col] > 0)
			{
				sum[bayer_type] += m_bayer_pixel[row*m_ImageWidth + col];
				count[bayer_type]++;
			}
		}
	}		
	*R = float(sum[BAYER_TYPE_R] / count[BAYER_TYPE_R]);
	*Gr = float(sum[BAYER_TYPE_Gr] / count[BAYER_TYPE_Gr]);
	*Gb = float(sum[BAYER_TYPE_Gb] / count[BAYER_TYPE_Gb]);
	*B = float(sum[BAYER_TYPE_B] / count[BAYER_TYPE_B]);

	return;	
}*/




void CImageObject::SaveToBMP(char *filename)
{
	BITMAPFILEHEADER bmFileHeader;
	BITMAPINFOHEADER bmInfoHeader;
	
	// BMP 헤더
	bmFileHeader.bfSize  = sizeof(BITMAPFILEHEADER);
	bmFileHeader.bfType  = 0x4D42;
	bmFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	
	bmInfoHeader.biSize    = sizeof(BITMAPINFOHEADER);
	bmInfoHeader.biWidth   = m_ImageWidth; // 너비
	bmInfoHeader.biHeight   = m_ImageHeight; // 높이
	bmInfoHeader.biPlanes   = 1;
	bmInfoHeader.biBitCount   = 24; // 색상 비트
	bmInfoHeader.biCompression = 0;
	bmInfoHeader.biSizeImage = bmInfoHeader.biWidth * bmInfoHeader.biHeight * (bmInfoHeader.biBitCount/8);
	
	bmInfoHeader.biXPelsPerMeter = 0;
	bmInfoHeader.biYPelsPerMeter = 0;
	bmInfoHeader.biClrUsed = 0;
	bmInfoHeader.biClrImportant = 0;

	BYTE *PixelFlip = new BYTE[m_ImageWidth * m_ImageHeight * 3];
	for (int row=0;row<m_ImageHeight;row++)
		for (int col=0;col<m_ImageWidth;col++)		
			CopyMemory(&PixelFlip[((m_ImageHeight - row - 1)*m_ImageWidth + col)*3], &m_rgb_pixel[(row*m_ImageWidth+col)*3], 3);
		
	CFile file;
	file.Open(filename, CFile::modeNoTruncate | CFile::modeCreate | CFile::modeReadWrite );
	file.Write(&bmFileHeader, sizeof(BITMAPFILEHEADER));
	file.Write(&bmInfoHeader, sizeof(BITMAPINFOHEADER));
	file.Write(PixelFlip, bmInfoHeader.biSizeImage);
	file.Close();

	delete [] PixelFlip;
}


void CImageObject::Duplicate( CImageObject &obj )
{
	m_ImageWidth = obj.m_ImageWidth;
	m_ImageHeight = obj.m_ImageHeight;
	m_ColorChannel = obj.m_ColorChannel;

	if (m_ImageHeight < 0 || m_ImageWidth < 0) return;
	SAFE_DELETE(m_rgb_pixel);
	m_rgb_pixel = new BYTE[m_ImageWidth*m_ImageHeight*3];
	memcpy(m_rgb_pixel, obj.m_rgb_pixel, m_ImageWidth*m_ImageHeight*3);



	// BMP 는 Bayer Data를 저장할 필요가 없겠쥐.
	/*if (obj.ImageType == IMAGE_TYPE_BMP) return;
	// 10 비트이면, 두 바이트 
	SAFE_DELETE(m_bayer_pixel);
	if (obj.m_bayer_pixel_bit_count == BAYER_PIXEL_BITS_8)
	{
		m_bayer_pixel = new BYTE[m_ImageWidth*m_ImageHeight];
		memcpy(m_bayer_pixel, obj.m_bayer_pixel, m_ImageWidth*m_ImageHeight);
	}
	else
	{
		m_bayer_pixel = new BYTE[m_ImageWidth*m_ImageHeight * 2];
		memcpy(m_bayer_pixel, obj.m_bayer_pixel, m_ImageWidth*m_ImageHeight * 2);
	}		
	SAFE_DELETE(m_bayer_type);
	m_bayer_type = new BYTE[m_ImageWidth*m_ImageHeight];
	memcpy(m_bayer_type, obj.m_bayer_type, m_ImageWidth*m_ImageHeight);*/
}

// Make BMP Image with 3 channel RGB buffer
void CImageObject::LoadBMPImageFromBuffer(BYTE *pPixel, int width, int height)
{
	SAFE_DELETE(m_rgb_pixel);
	m_rgb_pixel = new BYTE[width*height*3];
	memcpy(m_rgb_pixel, pPixel, width*height*3);
}
