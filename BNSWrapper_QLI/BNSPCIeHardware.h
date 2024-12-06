// BNSPCIeHardware.h    defeinitions that carry from application down to hardware
#pragma once
#ifndef BNSPCIeHARDWARE
#define BNSPCIeHARDWARE

#define BASE_TIME 100e-9			// the basic time increment board periods are measured in
#define BOARD_NAME_SIZE		20

//#include <afxwin.h>         // MFC core and standard components
//#include <afxext.h>         // MFC extensions
//#include <afxcmn.h>			// MFC support for Windows Common Controls

typedef struct {
	char 		    Name[20];	// should be a board name define constant	
	unsigned short	FrameWidth;
	unsigned short  FrameHeight;
	unsigned short	FrameDepth;		
	unsigned long	TimeIncrement;	// in 100ns
	unsigned long	FineTimeIncrement;
	int				DeviceNum;
	int				NumDevices;
} SBoard_Spec;


#endif  // BNSHARDWARE