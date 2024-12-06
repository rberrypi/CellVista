#include "stdafx.h"
#if SLM_PRESENT_BNS_ANCIENT == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "bns_device_old.h"
#include "qli_runtime_error.h"
auto const static dllname = "Interface.dll";

#define CALLING __cdecl
typedef void(CALLING* slm_constructor)(int, int);
typedef void(CALLING* slm_deconstructor)();
typedef void(CALLING* slm_set_download_mode)(bool);
typedef void(CALLING* slm_set_run_param)(int, int, int, int);
typedef void(CALLING* slm_read_lut_file)(unsigned char*, const char*); //docs are wrong
typedef void(CALLING* slm_write_frame_buffer)(int, unsigned char*, int);//docs are wrong
typedef void(CALLING* slm_select_image)(int);
typedef void(CALLING* slm_slm_power)(bool);
typedef void(CALLING* slm_set_run_mode)(char*, int, unsigned short*);
typedef int(CALLING* slm_get_run_status)();
HINSTANCE h_dll;               // Handle to DLL
slm_constructor constructor;
slm_deconstructor deconstructor;
slm_set_download_mode set_download_mode;
slm_set_run_param set_run_param;
slm_read_lut_file read_lut_file;
slm_write_frame_buffer write_frame_buffer;
slm_select_image select_image;
slm_slm_power slm_power;
slm_set_run_mode set_run_mode;
slm_get_run_status get_run_status;

bns_device_old::bns_device_old() : slm_device(512, 512, false)
{
	h_dll = LoadLibraryA(dllname);
	if (h_dll == nullptr)
	{
		qli_runtime_error("Can't find old BNS dll");
	}
	constructor = reinterpret_cast<slm_constructor>(GetProcAddress(h_dll, "Constructor"));
	deconstructor = reinterpret_cast<slm_deconstructor>(GetProcAddress(h_dll, "Deconstructor"));
	set_download_mode = reinterpret_cast<slm_set_download_mode>(GetProcAddress(h_dll, "SetDownloadMode"));
	set_run_param = reinterpret_cast<slm_set_run_param>(GetProcAddress(h_dll, "SetRunParam"));
	read_lut_file = reinterpret_cast<slm_read_lut_file>(GetProcAddress(h_dll, "ReadLUTFile"));
	write_frame_buffer = reinterpret_cast<slm_write_frame_buffer>(GetProcAddress(h_dll, "WriteFrameBuffer"));
	select_image = reinterpret_cast<slm_select_image>(GetProcAddress(h_dll, "SelectImage"));
	slm_power = reinterpret_cast<slm_slm_power>(GetProcAddress(h_dll, "SLMPower"));
	set_run_mode = reinterpret_cast<slm_set_run_mode>(GetProcAddress(h_dll, "SetRunMode"));
	get_run_status = reinterpret_cast<slm_get_run_status>(GetProcAddress(h_dll, "GetRunStatus"));
	const auto lc_type = 1;
	//
	const auto true_frames = 3;
	constructor(lc_type, true_frames);
	set_download_mode(false);
	//SetRunParam?
	unsigned char unused_lut[256] = { 0 };
	const auto linear = "linear.lut";
	read_lut_file(unused_lut, linear);
	//std::vector<unsigned char> blank(512 * 512, 0);
	//loadFrame(blank, 0);//required
	slm_power(true);
}


void bns_device_old::set_frame_internal(const int frame_number)
{
	select_image(frame_number);

}

void bns_device_old::load_frame_internal(const int num)
{
	const auto checked_size = 512;
	const auto data = frame_data_.at(num).data.data();
	write_frame_buffer(num, data, checked_size);
	set_frame_internal(num);//really why?
}
bns_device_old::~bns_device_old()
{
	slm_power(false);
	deconstructor();
}
#endif