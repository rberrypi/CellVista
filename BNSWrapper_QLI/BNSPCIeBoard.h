// BNSPCIeBoard.h   header file to declare CBNSFactory for generating various type of board objects
//				and the base class BNSPCIeBoard
#pragma once
#ifndef  BNSPCIeBOARD
#define BNSPCIeBOARD
//Moved this line
#undef AFX_DATA
#define AFX_DATA AFX_EXT_DATA

#include "BNSPCIeHardware.h"

// This class is exported from the BNSPCIeBoard.dll
// ReSharper disable CppInconsistentNaming
class AFX_EXT_CLASS CBNSPCIeBoard 
{
private:
	SBoard_Spec		m_BoardSpec;

public:
	virtual SBoard_Spec		*BoardSpec();

public:
	CBNSPCIeBoard();
	virtual ~CBNSPCIeBoard();

	virtual int WriteFrameBuffer(unsigned char *buffer);
	virtual int WriteLUT(unsigned char LUT[256]);

	virtual int SetTrueFrames(unsigned short TF);
	virtual int GetTrueFrames();

	virtual int SetLCType(bool bNematic);

	virtual int GetRunStatus();

	virtual int SetPower(bool powerOn);
	virtual int GetPower();
};


class AFX_EXT_CLASS CBNSPCIeFactory
{
public:
	CBNSPCIeFactory();
	~CBNSPCIeFactory();
#define nonsense CStringA
	CBNSPCIeBoard *BuildBoard(nonsense rqstBoardName, 
							  char buffer[_MAX_PATH], 
							  char buffer2[_MAX_PATH], 
							  bool bInit, 
							  bool TestEnable, 
							  bool RAMWriteEnable, 
							  bool *VerifyHardware);
};

#endif
// ReSharper restore CppInconsistentNaming


#undef AFX_DATA
#define AFX_DATA