#include "StdAfx.h"
//#include "Common.h"
#include <share.h>
#include <numeric>
#include "Common.h"


CLog::~CLog()
{
	if (m_file!= NULL)
	{
		fclose(m_file);
		//MessageBox(NULL, m_filename, "Warning", NULL);
	}
}

CLog::CLog( char *filename )
{
	m_file = NULL;
	m_header = false;
	strcpy(m_filename, filename);
	OpenFileSafely(filename);
}

CLog::CLog( char *filename, BOOL overWrite )
{
	m_file = NULL;
	strcpy(m_filename, filename);
	DeleteFile((LPCTSTR)filename);
	OpenFileSafely(filename);
}


bool CLog::OpenFileSafely( char *filename )
{
	if (m_file == NULL)
	{
		m_file = _fsopen(filename,"a+",_SH_DENYNO);
		if (m_file == NULL)
		{				
			char str[1024];
			sprintf(str, "%s alread opened in Excel", filename);

			//MessageBox(NULL, str, "Warning", NULL);
			return false;
		}
		fseek(m_file,0,SEEK_END);			
		return true;
	}
	return true;
}

void CLog::WriteHeader(const char *str )
{
	if (GetLineCount() == 0)
	{
		WriteLog(str);
	}
}

void CLog::WriteLog( LPCSTR format,... )
{

	if (m_file == NULL) return;
	char szMes[10240];
	va_list args;
	va_start(args,format);
	vsprintf(szMes,format,args); 
	va_end(args);

	//if (!OpenFileSafely(m_filename)) return;

	fprintf(m_file, szMes);
}

int CLog::GetLineCount()
{
	FILE *fp;
	fp = _fsopen(m_filename, "r", _SH_DENYNO);
	if (fp == 0) return 0;

	char text[1024];
	int count = 0;
	while (fgets(text, 1023, fp))
	{
		count++;
	}
	fclose(fp);
	return count;
}


CLogEx::CLogEx( char *filename )
{
	m_file = NULL;
	m_header = false;
	strcpy(m_filename, filename);
	OpenFileSafely(filename);
}

CLogEx::CLogEx( char *filename, BOOL overWrite )
{
	m_file = NULL;
	strcpy(m_filename, filename);
	DeleteFile((LPCSTR)filename);
	OpenFileSafely(filename);
}

void CLogEx::AddLogItem( char  *name, char * format,... )
{
	if (m_file == NULL) return;
	char szMes[1024];
	va_list args;
	va_start(args,format);
	vsprintf(szMes,format,args); 
	va_end(args);

	//if (!OpenFileSafely(m_filename)) return;

	//fprintf(m_file, szMes);

	vecLogItem.push_back(CLogItem(name, szMes));
}

void CLogEx::Flush()
{
	char str[10240] = {""};

	if (GetLineCount() == 0)
	{
		for (int i=0;i<(int)vecLogItem.size();i++)
		{
			strcat(str, vecLogItem[i].name);
			strcat(str, ",");
		}
		strcat(str, "\n");


		WriteHeader(str);
	}

	for (int i=0;i<(int)vecLogItem.size();i++)
	{			

		WriteLog("%s,", vecLogItem[i].str);	

	}
	WriteLog("\n");
}


CMatLoader::CMatLoader( char *name )
{
	CStdioFile file;
	char filename[100];
	sprintf(filename, "Binary\\%s.bin", name);
	if (!file.Open(filename,CFile::modeRead | CFile::typeBinary)) return;

	//file.Open("binary\\" + name + ".bin",CFile::modeRead | CFile::typeBinary);

	int dimCount;
	file.Read(&dimCount, sizeof(int));

	dim.resize(dimCount);

	file.Read(&dim.front(), sizeof(int) * dimCount);

	int totalSize = 1;
	//wcout << name.GetString() <<  "  [ ";
	vector <int>::iterator iter;
	for (iter = dim.begin();iter != dim.end();iter++)
	{
		totalSize *= *iter;
		//if (iter != dim.begin())
		//	cout << " x ";
		//cout << *iter;
	}

	//cout << " ]" << endl;

	data = new double[totalSize];
	// file.Read(&data, sizeof(double) * totalSize);
	file.Read(data, sizeof(double) * totalSize);

	double checksum = accumulate(data, data + totalSize , 0.0);

	file.Close();
}

CMatLoader::CMatLoader( char *name, char *path)
{
	CStdioFile file;
	char filename[100];
	sprintf(filename, "%s\\%s.bin", path, name);
	
	if (!file.Open(filename,CFile::modeRead | CFile::typeBinary)) 
	{		
		return;
	}


	int dimCount;
	file.Read(&dimCount, sizeof(int));

	dim.resize(dimCount);

	file.Read(&dim.front(), sizeof(int) * dimCount);

	int totalSize = 1;
	//wcout << name.GetString() <<  "  [ ";
	vector <int>::iterator iter;
	for (iter = dim.begin();iter != dim.end();iter++)
	{
		totalSize *= *iter;
		//if (iter != dim.begin())
		//	cout << " x ";
		//cout << *iter;
	}

	//cout << " ]" << endl;

	data = new double[totalSize];
	// file.Read(&data, sizeof(double) * totalSize);
	file.Read(data, sizeof(double) * totalSize);

	double checksum = accumulate(data, data + totalSize , 0.0);

	file.Close();
}


CMatLoader::~CMatLoader()
{
	SAFE_DELETE(data);
}

void CMatLoader::Seriallize()
{
	int size = 1;
	int i;
	for (i=0;i<(int)dim.size();i++)
		size *= dim[i];

	dim.resize(2);
	dim[0] = size;
	dim[1] = 1;
}
