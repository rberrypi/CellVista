// BNSPCIeSDKDlg.h : header file
//

#pragma once
//Don't curse! ~MK
//what the hell si code on the button
#if !defined(AFX_BNSPCIeSDKDlg_H__023C5C2A_4888_4F4A_9053_00875965D4E9__INCLUDED_)
#define AFX_BNSPCIeSDKDlg_H__023C5C2A_4888_4F4A_9053_00875965D4E9__INCLUDED_

#include "BNSPCIeBoard.h"

typedef struct
{
	//pointer to the board
	CBNSPCIeBoard* theBoard;

	unsigned char*  FrameOne;
	unsigned char*  FrameTwo;
	unsigned char*	PhaseCompensationData;
	unsigned char*  SystemPhaseCompensationData;
	unsigned char*	LUT;

	CString			LUTFileName;
	CString			PhaseCompensationFileName;
	CString			SystemPhaseCompensationFileName;
} Board_Entry;

typedef CTypedPtrList<CPtrList, Board_Entry*> CBoard_List;

/////////////////////////////////////////////////////////////////////////////
// CBNSPCIeSDKDlg dialog
// Converted to a static utility ~MK
class CBNSPCIeSDKDlg   /*Should it inheret CDialoag?*/
{
// Construction
public:
	//Functions
	//static bool			ReadLUTFile(unsigned char* LUTBuf, CString LUTPath);
	static bool	 setLinearLUT( unsigned char *LUTBuf);

	static int			LC_Type;
	static int			FrameRate;
	static unsigned short TrueFrames;

	static int			ImgHeight;
	static int			ImgWidth;
	static bool	m_CompensatePhase;
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

/*
int CBNSPCIeSDKDlg::LC_Type = 0;
int CBNSPCIeSDKDlg::FrameRate = 0;
unsigned short CBNSPCIeSDKDlg::TrueFrames = 0;
int CBNSPCIeSDKDlg::ImgHeight = 0;
int CBNSPCIeSDKDlg::ImgWidth = 0;
bool CBNSPCIeSDKDlg::m_CompensatePhase = 0;
//*/

#endif // !defined(AFX_BNSPCIeSDKDlg_H__023C5C2A_4888_4F4A_9053_00875965D4E9__INCLUDED_)
