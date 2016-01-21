#pragma once

#include <vector>
#include <windows.h>
#define BOOL int


using namespace std;

#define SAFE_DELETE(ptr) if (ptr != NULL) delete [] ptr;
#define ROUND(x) int((x) + 0.5)



class CLogItem
{
public:
	char name[100];	
	char str[1000];

	CLogItem(char *lname, char *lstr)
	{
		strcpy(name, lname);
		strcpy(str, lstr);
	};
};

class CLog
{
protected:
	FILE *m_file;
	char m_filename[100];
	bool m_header;


public:
	CLog() {};	
	~CLog();;

	CLog(char *filename);

	CLog(char *filename, BOOL overWrite);;

	bool OpenFileSafely(char *filename);

	void WriteHeader(const char *str);

	void WriteLog(const char *format,...);

	int GetLineCount();
};


class CLogEx : public CLog
{
private:

	vector <CLogItem> vecLogItem;
public:
	CLogEx() {};
	CLogEx(char *filename);

	CLogEx(char *filename, BOOL overWrite);


	// do not use with AddLogItem
	void AddLogItem(char * name, char  *format,...);


	void Flush();

};



class CRectEx : public RECT {
public:

	//int left, top, right, bottom;

	CRectEx()
	{
		left = 0; top = 0; right = 0; bottom = 0;
	};

	CRectEx(int l, int t, int r, int b)
	{
		left = l;
		top = t;
		right = r;
		bottom = b;
	};

	inline int GetArea() {
		return Width() * Height();
	};

	inline int Width() {
		return right - left + 1;
	};

	inline int Height() {
		return bottom - top + 1;
	};

	inline void Offset(int x, int y)
	{
		left += x;
		right += x;
		top += y;
		bottom += y;
	}

	inline void SetRect(int lleft, int ltop, int lright, int lbottom)
	{
		left = lleft;
		top = ltop;
		right = lright;
		bottom = lbottom;
	}

	inline void SetRectCentered(int centerx, int centery, int width, int height)
	{
		left = centerx - (width / 2);
		top = centery - (height / 2);
		right = left + width - 1;
		bottom = top + height - 1;
	}

	inline BOOL EvaluateRect(int width, int height)
	{
		if (left < 0) return FALSE;
		if (top < 0) return FALSE;
		if (right > width ) return FALSE;
		if (bottom > height) return FALSE;
		return TRUE;
	};
	inline POINT CenterPoint()
	{
		POINT center;
		center.x = (left + right) / 2;
		center.y = (top + bottom) / 2;
		return center;
	};

	// POINT가 rect 안에 드는 것인지 검사해서 TRUE 리턴 
	inline BOOL PointInRect(int x, int y)
	{
		if (x >= left && x <= right)
			if (y >= top && y <= bottom)
				return TRUE;
		return FALSE;
	};
};




class CStopWatch
{
private:
	LARGE_INTEGER freq, start, end;
public:
	CStopWatch()
	{
		QueryPerformanceFrequency(&freq);
		start.QuadPart = 0; end.QuadPart = 0;
		StartTime();
	};

	void StartTime()
	{		
		QueryPerformanceCounter(&start);
	};

	double CheckTime()
	{
		QueryPerformanceCounter(&end);

		return (double)(end.QuadPart-start.QuadPart)/freq.QuadPart*1000;
		//TRACE("%.2f msec\n", elapsed);
	};
};



class CMatLoader
{
public :
	vector <int> dim;
	double *data;
	CMatLoader() {data = NULL;};
	void Create(int m, int n) {
		dim.push_back(m); dim.push_back(n);
		data = new double[m*n];
		ZeroMemory(data, sizeof(double) * m*n);
	};
	~CMatLoader();
	CMatLoader(char * name);
	CMatLoader(char *name, char *path);
	void Seriallize();

};
