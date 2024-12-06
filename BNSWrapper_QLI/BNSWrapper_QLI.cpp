// BNSWrapper_QLI.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "BNSWrapper_QLI.h"
#include "BNSPCIeSDKDlg.h" // mystery class magic
#include "bnsHacks.h"
#include <cassert>
// This is the constructor of a class that has been exported.
// see BNSWrapper_QLI.h for the class definition
#include <array>
CBNSWrapper_QLI::CBNSWrapper_QLI() :handle(nullptr)
{
	CBNSPCIeFactory board_factory;
	std::array<char, _MAX_PATH> buffer={0};
	std::array<char, _MAX_PATH> buffer1={0};
	const CStringA board_name = "PCIe512";
	auto verify_hardware = true;
	const auto test_enable = false;
	const auto ram_write_enable = false; // try true
	//open communication with our first PCIe board
	handle = new Board_Entry;
	static auto first = 0;
	static_cast<Board_Entry*>(handle)->theBoard = board_factory.BuildBoard(board_name, buffer.data(), buffer1.data(), ((first++) == 0), test_enable, ram_write_enable, &verify_hardware); //good
	assert(verify_hardware);//Correct hardware :-)
	//pBoard->theBoard->SetPower(true);
	static_cast<Board_Entry*>(handle)->PhaseCompensationFileName.Format(L"BLANK.BMP");  //good (maybe should be generated?
	static_cast<Board_Entry*>(handle)->SystemPhaseCompensationFileName.Format(L"BLANK.BMP");  //good
	const auto img_width = static_cast<Board_Entry*>(handle)->theBoard->BoardSpec()->FrameWidth;
	const auto img_height = static_cast<Board_Entry*>(handle)->theBoard->BoardSpec()->FrameHeight;
	CBNSPCIeSDKDlg::ImgHeight = img_height;
	CBNSPCIeSDKDlg::ImgWidth = img_width;
	static_cast<Board_Entry*>(handle)->LUT = new unsigned char[256];//good
	bnsSafeCall(CBNSPCIeSDKDlg::setLinearLUT((static_cast<Board_Entry*>(handle))->LUT));  //good
	static_cast<Board_Entry*>(handle)->theBoard->WriteLUT(static_cast<Board_Entry*>(handle)->LUT); //good
#define nematic 1
#if nematic == 1
	{
		static_cast<Board_Entry*>(handle)->theBoard->SetLCType(true);
	}
#else
	{
		static_cast<Board_Entry*>(handle)->theBoard->SetLCType(false);
		memset(static_cast<Board_Entry*>(handle)->PhaseCompensationData, 0, img_height*img_width);
		memset(static_cast<Board_Entry*>(handle)->SystemPhaseCompensationData, 0, img_height*img_width);
		//float TimeIncrement = static_cast<float>((static_cast<Board_Entry*>(handle)->theBoard->BoardSpec()->TimeIncrement)*1e-7);
		//auto fTrueFrames = static_cast<float>(1.0 / (1000 * 2.0*TimeIncrement));
	}
#endif
	static_cast<Board_Entry*>(handle)->theBoard->SetTrueFrames(3);
}

CBNSWrapper_QLI::~CBNSWrapper_QLI()
{
	auto p_board = static_cast<Board_Entry*>(handle);
	p_board->theBoard->SetPower(false);
	delete p_board->theBoard;
	delete p_board;
}

void CBNSWrapper_QLI::set_pattern( unsigned char* in) const
{
	assert(in);
	auto p_board = static_cast<Board_Entry*>(handle);
	p_board->theBoard->WriteFrameBuffer(in);// Is this asynchronous?
}
