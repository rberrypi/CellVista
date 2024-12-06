#include "stdafx.h"
#if (SLM_PRESENT_MEADOWLARK_RETARDER == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS) || (SLM_PRESENT_MEADOWLARK_HS_RETARDER == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS)
#define NOMINMAX
#include "meadowlark_retarder.h"
#include "usbdrvd.h"
#include <iostream>
#include "time_slice.h"
#include "qli_runtime_error.h"
#pragma comment( lib, "usbdrvd.lib" )
#define  flagsandattrs  0x40000000

meadowlark_retarder::meadowlark_retarder() : slm_device(1, 1, true)
{
	std::lock_guard<std::recursive_mutex> lk(command_issuing_mutex);
	UINT devcnt;
	UINT USB_PID;
	GUID  theGUID;
	theGUID.Data1 = 0xa22b5b8b;
	theGUID.Data2 = 0xc670;
	theGUID.Data3 = 0x4198;
	theGUID.Data4[0] = 0x93;
	theGUID.Data4[1] = 0x85;
	theGUID.Data4[2] = 0xaa;
	theGUID.Data4[3] = 0xba;
	theGUID.Data4[4] = 0x9d;
	theGUID.Data4[5] = 0xfc;
	theGUID.Data4[6] = 0x7d;
	theGUID.Data4[7] = 0x2b;
#if SLM_PRESENT_MEADOWLARK_HS_RETARDER || BUILD_ALL_DEVICES_TARGETS
	USB_PID = 0x1437; //PID for the D5020 HS.
#endif
#if SLM_PRESENT_MEADOWLARK_RETARDER
	USB_PID = 0x139C; //PID for the D5020.
#endif
	devcnt = USBDRVD_GetDevCount(USB_PID);
	if (devcnt == 0)
	{
		std::cout << "No Meadowlark Optics USB Devices Present." << std::endl;
		return;
	}
	/* open device and pipes */
	dev1 = USBDRVD_OpenDevice(1, flagsandattrs, USB_PID);
	pipe0 = USBDRVD_PipeOpen(1, 0, flagsandattrs, &theGUID);
	pipe1 = USBDRVD_PipeOpen(1, 1, flagsandattrs, &theGUID);
	//
	issue_command("ver:?\n", true, 47);
}

void meadowlark_retarder::hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings)
{
	qli_not_implemented();
}

void meadowlark_retarder::issue_command(const std::string& command, const bool echo_result, const int check_size)
{

	std::lock_guard<std::recursive_mutex> lk(command_issuing_mutex);
	//time_slice ts("Issue Command");
	auto letters = 0;
	for (; letters < command.size(); ++letters)
	{
		conversion_buffer[letters] = command.at(letters);
	}
	const auto wrote_length = USBDRVD_BulkWrite(dev1, 1, conversion_buffer.data(), letters);
	if (wrote_length != letters)
	{
		qli_runtime_error("Meadowlark LCVR Communication Failure");
	}
	const auto amount_read = USBDRVD_BulkRead(dev1, 0, status_buffer.data(), status_buffer.size());
	if (check_size >= 0 && amount_read != check_size)
	{
		qli_runtime_error("Meadowlark LCVR Communication Failure");
	}
	if (echo_result)
	{
		/* read status response */
		std::cout << "Sent: " << command;
		std::cout << "Reply: ";
		for (auto i = 0; i < amount_read; ++i)
		{
			std::cout << status_buffer.at(i);
		}
		std::cout << std::endl;
	}
	std::fill(conversion_buffer.begin(), conversion_buffer.end(), 0);
	std::fill(status_buffer.begin(), status_buffer.end(), 0);
}


void meadowlark_retarder::load_frame_internal(int)
{
	//blank on purpose
}

float meadowlark_retarder::char_to_voltage(const unsigned char input, const float voltage_limit)
{
	//duplicated code
	constexpr auto min = 0;
	constexpr auto max = 255;
	constexpr auto a = 0.0;
	const auto b = voltage_limit;//volts
	const auto return_me = (b - a) * (input - min) / (max - min) + a;
	return return_me;
}

void meadowlark_retarder::set_frame_internal(const int num)
{
	const auto max_mili_volts = 10 * 1000.f;//10 V
	const auto stored_char = frame_data_.at(num).data.front();
	const auto as_volts = char_to_voltage(stored_char, voltage_max);
	const auto as_mili_volts = static_cast<int>(round(std::min(as_volts * 1000, max_mili_volts)));
	const auto str = "inv:1," + std::to_string(as_mili_volts) + "\n";
	const auto echo = true;
	issue_command(str, echo, -1);  
}



meadowlark_retarder::~meadowlark_retarder()
{
	std::lock_guard<std::recursive_mutex> lk(command_issuing_mutex);
	USBDRVD_PipeClose(pipe0);
	USBDRVD_PipeClose(pipe1);
	USBDRVD_CloseDevice(dev1);
}



#endif