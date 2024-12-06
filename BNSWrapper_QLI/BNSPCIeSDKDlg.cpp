/*
 * File from the SDK without the initialization
 * Mikhail Kandel 1/28/2013
 *
*/
// BNSPCIeSDKDlg.cpp : implementation file
//

#include "stdafx.h"
#include "BNSPCIeSDKDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
//static char THIS_FILE[] = __FILE__;
#endif

#define MAX_LINE 300


//////////////////////////////////////////////////////////////////////////////////
//
//  ReadLUTFile()
//
//  Inputs: the name of the LUT file to read, and an array to store the file in
//
//  Returns: true if read successfully, false if linear.lut was generated
//
//  Purpose: This will read in the LUT file. This is a look up table that we process 
//			 our images through. What it does is map the pixel values to the values
//			 specified by the LUT. For example with Linear.LUT we have a direct 
//			 mapping, so if the pixel value is 255, Linear.LUT will keep it at 255.
//			 However, skew.LUT will alter the pixel values. With skew.LUT if the 
//			 initial pixel value is 60, then it is mapped to a value of 139. 
//
//  Modifications:
//
/////////////////////////////////////////////////////////////////////////////////////

bool CBNSPCIeSDKDlg::setLinearLUT( unsigned char *LUTBuf)
{
	for (auto i=0;i<256;i++)
	{
		LUTBuf[i]=i;
	}
	/*
	range=180:240;
	input=(floor(imresize(range,[1,255])));
	for i=input
	fprintf('%u,\n',i);
	end
	
	const bool cropit = false; //On the BNS SLM this is purley a digital effect...
	if (cropit)
	{
		unsigned char cropped[255] = {
			179,
			179,
			180,
			180,
			180,
			180,
			181,
			181,
			181,
			181,
			182,
			182,
			182,
			182,
			182,
			183,
			183,
			183,
			183,
			184,
			184,
			184,
			184,
			185,
			185,
			185,
			185,
			186,
			186,
			186,
			186,
			187,
			187,
			187,
			187,
			187,
			188,
			188,
			188,
			188,
			189,
			189,
			189,
			189,
			190,
			190,
			190,
			190,
			191,
			191,
			191,
			191,
			192,
			192,
			192,
			192,
			193,
			193,
			193,
			193,
			193,
			194,
			194,
			194,
			194,
			195,
			195,
			195,
			195,
			196,
			196,
			196,
			196,
			197,
			197,
			197,
			197,
			198,
			198,
			198,
			198,
			198,
			199,
			199,
			199,
			199,
			200,
			200,
			200,
			200,
			201,
			201,
			201,
			201,
			202,
			202,
			202,
			202,
			203,
			203,
			203,
			203,
			204,
			204,
			204,
			204,
			204,
			205,
			205,
			205,
			205,
			206,
			206,
			206,
			206,
			207,
			207,
			207,
			207,
			208,
			208,
			208,
			208,
			209,
			209,
			209,
			209,
			210,
			210,
			210,
			210,
			210,
			211,
			211,
			211,
			211,
			212,
			212,
			212,
			212,
			213,
			213,
			213,
			213,
			214,
			214,
			214,
			214,
			215,
			215,
			215,
			215,
			215,
			216,
			216,
			216,
			216,
			217,
			217,
			217,
			217,
			218,
			218,
			218,
			218,
			219,
			219,
			219,
			219,
			220,
			220,
			220,
			220,
			221,
			221,
			221,
			221,
			221,
			222,
			222,
			222,
			222,
			223,
			223,
			223,
			223,
			224,
			224,
			224,
			224,
			225,
			225,
			225,
			225,
			226,
			226,
			226,
			226,
			226,
			227,
			227,
			227,
			227,
			228,
			228,
			228,
			228,
			229,
			229,
			229,
			229,
			230,
			230,
			230,
			230,
			231,
			231,
			231,
			231,
			232,
			232,
			232,
			232,
			232,
			233,
			233,
			233,
			233,
			234,
			234,
			234,
			234,
			235,
			235,
			235,
			235,
			236,
			236,
			236,
			236,
			237,
			237,
			237,
			237,
			237,
			238,
			238,
			238,
			238,
			239,
			239,
			239,
			239,
			240,
			240
		};
		auto bytes = sizeof(unsigned char) * 255;
		memcpy(LUTBuf, cropped, bytes);
	}
	*/
	return true;
}

int CBNSPCIeSDKDlg::LC_Type = 0;
int CBNSPCIeSDKDlg::FrameRate = 0;
unsigned short CBNSPCIeSDKDlg::TrueFrames = 0;
int CBNSPCIeSDKDlg::ImgHeight = 0;
int CBNSPCIeSDKDlg::ImgWidth = 0;
bool CBNSPCIeSDKDlg::m_CompensatePhase = false;