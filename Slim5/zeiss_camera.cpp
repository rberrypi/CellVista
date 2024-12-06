#include "stdafx.h"
#if CAMERA_PRESENT_ZEISSMR == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "time_slice.h"
#include "zeiss_camera.h"
#include "mcam_zei.h"
#include "write_tif.h"
#include <iostream>
#pragma comment(lib, "mrfw64.lib")
#define AXIO_SAFE_CALL(err) axio_safe_call(err,__FILE__,__LINE__)
void axio_safe_call(const long err, const char* file, const int line)
{
	if (err != NOERR)
	{
		std::cout << err << " @" << line << ":" << file << ": ";
		switch (err)
		{
		case DOWNLOADERR: std::cout << "Carl Zeiss AxioCam has not yet been successfully initialized." << std::endl; break;
		case  INITERR: std::cout << "Carl Zeiss AxioCam has not yet been successfully initialized." << std::endl; break;
		case  NOCAMERA: std::cout << "Carl Zeiss AxioCam not found! Please check cabling and power supply of the AxioCam." << std::endl; break;
		case  ABORTERR: std::cout << "Operation was aborted by the application.This happens for instance when McamAbortFastAcquisition is called." << std::endl; break;
		case  WHITEERR: std::cout << "White point too bright or too dark when trying to perform white balance." << std::endl; break;
		case  IMAGESIZEERR: std::cout << "Internal memory error(image size) happened during acquisition." << std::endl; break;
		case  NOMEMERR: std::cout << "Not enough memory available. Internal Error." << std::endl; break;
		case  PARAMERR: std::cout << "Invalid parameters." << std::endl; break;
		case  CAMERABUSY: std::cout << "Camera busy.You might have to abort the running camera operation by calling McamAbortFastAcquisition." << std::endl; break;
		case  CAMERANOTSTARTED: std::cout << "Camera not started." << std::endl; break;
		case  BLACKREFTOOBRIGHT: std::cout << "Image too bright while executing black reference.Please offer a completely dark image." << std::endl; break;
		case  WHITEREFTOOBRIGHT: std::cout << "Image too bright while executing white reference.Please reduce exposure time or illumination." << std::endl; break;
		case  WHITEREFTOODARK: std::cout << "Image too dark while executing white reference.Please increase exposure time or illumination." << std::endl; break;
		case  NOTIMPLEMENTED: std::cout << "The called function is not implemented for this camera hardware." << std::endl; break;
		case  NODEVICEFOUND: std::cout << "Device not found." << std::endl; break;
		case  HARDWAREVERSIONCONFLICT: std::cout << " Incorrect hardware revision. Please perform an AxioCam driver update to the current version." << std::endl; break;
		case  FIRMWAREVERSIONCONFLICT: std::cout << "Incorrect firmware revision. Please perform an AxioCam driver update to the current version." << std::endl; break;
		case  READERROR: std::cout << "Internal error for a Firewire AxioCam(code 20)." << std::endl; break;
		case  TRIGGERERROR: std::cout << "Internal error for a Firewire AxioCam(code 21)." << std::endl; break;
		case  BANDWIDTHERROR: std::cout << "The required bandwidth for transmitting camera images via the Firewire bus could not be allocated.To avoid this problem, reduce the traffic on the bus by enabling 8bit compression or by setting a(smaller) ROI." << std::endl; break;
		case  RESOURCEERROR: std::cout << "Internal error for a Firewire AxioCam(code 23)." << std::endl; break;
		case  ATTACHERROR: std::cout << "Internal error for a Firewire AxioCam(code 24)." << std::endl; break;
		case  CHANNELERROR: std::cout << "Internal error for a Firewire AxioCam(code 25)." << std::endl; break;
		case  STOPERROR: std::cout << "Internal error for a Firewire AxioCam(code 26)." << std::endl; break;
		case  WRITEERROR: std::cout << "Internal error for a Firewire AxioCam(code 27)." << std::endl; break;
		case  EPROMERR: std::cout << "Internal error for a Firewire AxioCam(code 28)." << std::endl; break;
		case  BUSRESETERR: std::cout << "Internal error for a Firewire AxioCam(code 29)." << std::endl; break;
		default: break;
		}
		qli_runtime_error("Zeiss Error");
	}
}

struct zeiss_camera_imp : boost::noncopyable  // NOLINT(hicpp-special-member-functions)
{
	HMODULE m_h_library;
	long (WINAPI* p_mcamm_info)(long cameraindex, SMCAMINFO* p_info);

	// Resolution
	long (WINAPI* p_mcamm_get_current_resolution)(long cameraindex);
	long (WINAPI* p_mcamm_get_number_of_resolutions)(long cameraindex);
	long (WINAPI* p_mcamm_get_resolution_values)(long cameraindex, long index, long* p_width, long* p_height, eMcamScanMode* p_mode);
	long (WINAPI* p_mcamm_set_resolution)(long cameraindex, long index);

	// Colordepth
	long (WINAPI* p_mcamm_get_current_bits_per_pixel)(long cameraindex);
	BOOL(WINAPI* p_mcamm_has_bits_per_pixel)(long cameraindex, long bpp, BOOL* bhasbits);
	long (WINAPI* p_mcamm_set_bits_per_pixel)(long cameraindex, long bpp);

	// Binning
	long (WINAPI* p_mcamm_get_current_binning)(long cameraindex, long* binning);
	BOOL(WINAPI* p_mcamm_has_binning)(long cameraindex, long binning, BOOL* bhasbinning);
	long (WINAPI* p_mcamm_set_binning)(long cameraindex, long binning);

	// Region Of Interest
	long (WINAPI* p_mcamm_get_currentframe_size)(long cameraindex, RECT* p_rect);
	BOOL(WINAPI* p_mcamm_has_frame)(long cameraindex, BOOL* bhasframe);
	long (WINAPI* p_mcamm_set_frame_size)(long cameraindex, RECT* p_rect);

	// Readout Time (McammGetCurrentReadouttime)
	long (WINAPI* p_mcamm_get_current_readout_time)(long cameraindex, long* time);

	// Exposure Time
	long (WINAPI* p_mcamm_get_current_exposure)(long cameraindex, long* microseconds);
	long (WINAPI* p_mcammget_exposure_range)(long cameraindex, long* pmin, long* pmax, long* pinc);
	long (WINAPI* p_mcamm_set_exposure)(long cameraindex, long newval);

	// acquisition
	long (WINAPI* p_mcammacquisition)(long cameraindex, unsigned short* ptr, long size, McamImageProcEx cb);
	long (WINAPI* p_mcammacquisition_ex)(long cameraindex, unsigned short* ptr, long size, McamImageProcEx cb, void* user_param);
	long (WINAPI* p_mcamm_start_fastacquisition)(long cameraindex);
	long (WINAPI* p_mcamm_abort_fastacquisition)(long cameraindex);
	long (WINAPI* p_mcamm_is_fastacquisition_ready)(long cameraindex, unsigned short* p_image_data, long allocated_size, BOOL b_start_next);
	long (WINAPI* p_mcamm_get_current_data_size)(long cameraindex, long* p_width, long* p_height);

	// White Balance
	long (WINAPI* p_mcamm_get_current_white_balance)(long cameraindex, double* p_red, double* p_green, double* p_blue);
	long (WINAPI* p_mcamm_set_image_white_balance)(long cameraindex, double red, double green, double blue);
	long (WINAPI* p_mcamm_set_white_balance)(long cameraindex, double red, double green, double blue);

	// White Reference (Shading Correction)
	long (WINAPI* p_mcamm_calculate_white_ref_ex)(long cameraindex, McamImageProcEx p_call_back, void* user_param);
	BOOL(WINAPI* p_mcamm_get_current_white_ref)(long cameraindex, BOOL* bgetwref);
	BOOL(WINAPI* p_mcamm_has_white_ref)(long cameraindex, BOOL* bhaswref);
	long (WINAPI* p_mcamm_set_white_ref)(long cameraindex, BOOL b_enable);
	long (WINAPI* p_mcamm_get_white_table)(long cameraindex, short* table);
	long (WINAPI* p_mcamm_set_white_table)(long cameraindex, short* table);

	// Black Reference
	long (WINAPI* p_mcamm_calculate_black_ref_ex)(long cameraindex, McamImageProcEx p_call_back, void* user_param);
	BOOL(WINAPI* p_mcamm_get_current_black_ref)(long cameraindex, BOOL* bgetbref);
	BOOL(WINAPI* p_mcamm_has_black_ref)(long cameraindex, BOOL* bhasbref);
	long (WINAPI* p_mcamm_set_black_ref)(long cameraindex, BOOL b_enable);
	long (WINAPI* p_mcamm_restore_black_ref)(long cameraindex, unsigned short* ref, long bytesize);
	long (WINAPI* p_mcamm_save_black_ref)(long cameraindex, unsigned short* ref, long bytesize);

	// Filter Operations
	BOOL(WINAPI* p_mcamm_is_color_processing_enabled)(long cameraindex, BOOL* bisenabled);
	void (WINAPI* p_mcamm_enable_color_processing)(long cameraindex, BOOL b_enable);
	long (WINAPI* p_mcamm_set_blooming_voltage)(long cameraindex, long volt);         // MR only
	long (WINAPI* p_mcamm_get_blooming_voltage)(long cameraindex, long* volt);

	// Trigger In/Out
	long (WINAPI* p_mcamm_enable_hardware_trigger)(long cameraindex, BOOL b_enable);
	long (WINAPI* p_mcamm_set_hardware_trigger_polarity)(BOOL b_enable);
	long (WINAPI* p_mcamm_is_hardware_trigger_enabled)(long cameraindex, BOOL* pb_enable);
	long (WINAPI* p_mcamm_get_current_shutter_delay)(long cameraindex, long* open_delay, long* close_delay);
	long (WINAPI* p_mcamm_set_shutter_delay)(long cameraindex, long open_delay, long close_delay);
	long (WINAPI* p_mcamm_shutter_control)(long cameraindex, BOOL b_auto, BOOL b_invert);

	// Continuous Mode
	long (WINAPI* p_mcamm_get_ip_info)(long cameraindex, void** p_context, unsigned long* p_context_byte_size, ContinuousCallbackProc cbproc, unsigned long* p_img_byte_size);
	long (WINAPI* p_mcamm_start_continuousacquisition)(long cameraindex, long thread_priority, void* user_param);
	long (WINAPI* p_mcamm_stop_continuousacquisition)(long cameraindex);
	long (WINAPI* p_mcamm_execute_ip_function)(void* p_context, unsigned short* p_image_data, unsigned short* p_image_data_to_process);

	//Prepared acquisition
	long (WINAPI* p_mcamm_initializeacquisition)(long cameraindex);
	long (WINAPI* p_mcamm_finalizeacquisition)(long cameraindex);
	long (WINAPI* p_mcamm_nextacquisition)(long cameraindex, unsigned short* p_image_data, long allocated_size, long microseconds);

	long (WINAPI* p_close_mr3)();
	long (WINAPI* p_init_mr3)();

	void load_functions()
	{
		// Camera Initialization & Status
		p_mcamm_info = reinterpret_cast<long (WINAPI*)(long, SMCAMINFO*)>(GetProcAddress(m_h_library, "McammInfo"));
		assert(p_mcamm_info);
		// Resolution
		p_mcamm_get_current_resolution = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammGetCurrentResolution"));
		assert(p_mcamm_get_current_resolution);
		p_mcamm_get_number_of_resolutions = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammGetNumberOfResolutions"));
		assert(p_mcamm_get_number_of_resolutions);
		p_mcamm_get_resolution_values = reinterpret_cast<long (WINAPI*)(long, long, long*, long*, eMcamScanMode*)>(GetProcAddress(
			m_h_library, "McammGetResolutionValues"));
		assert(p_mcamm_get_resolution_values);
		p_mcamm_set_resolution = reinterpret_cast<long (WINAPI*)(long, long)>(GetProcAddress(m_h_library, "McammSetResolution"));
		assert(p_mcamm_set_resolution);
		// Continuous Mode
		p_mcamm_get_ip_info = reinterpret_cast<long (WINAPI*)(long, void**, unsigned long*, ContinuousCallbackProc, unsigned long*)>(
			GetProcAddress(m_h_library, "McammGetIPInfo"));
		assert(p_mcamm_get_ip_info);
		p_mcamm_start_continuousacquisition = reinterpret_cast<long (WINAPI*)(long, long, void*)>(GetProcAddress(m_h_library, "McammStartContinuousAcquisition"));
		assert(p_mcamm_start_continuousacquisition);
		p_mcamm_stop_continuousacquisition = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammStopContinuousAcquisition"));
		assert(p_mcamm_stop_continuousacquisition);
		p_mcamm_execute_ip_function = reinterpret_cast<long (WINAPI*)(void*, unsigned short*, unsigned short*)>(GetProcAddress(
			m_h_library, "McammExecuteIPFunction"));
		assert(p_mcamm_execute_ip_function);
		//Prepared acquisition
		p_mcamm_initializeacquisition = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammInitializeAcquisition"));
		assert(p_mcamm_initializeacquisition);
		p_mcamm_nextacquisition = reinterpret_cast<long (WINAPI*)(long, unsigned short*, long, long)>(GetProcAddress(
			m_h_library, "McammNextAcquisition"));
		assert(p_mcamm_nextacquisition);
		p_mcamm_finalizeacquisition = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammFinalizeAcquisition"));
		assert(p_mcamm_finalizeacquisition);
		// Colordepth
		p_mcamm_get_current_bits_per_pixel = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammGetCurrentBitsPerPixel"));
		assert(p_mcamm_get_current_bits_per_pixel);
		p_mcamm_has_bits_per_pixel = reinterpret_cast<int (WINAPI*)(long, long, BOOL*)>(GetProcAddress(m_h_library, "McammHasBitsPerPixel"));
		assert(p_mcamm_has_bits_per_pixel);
		p_mcamm_set_bits_per_pixel = reinterpret_cast<long (WINAPI*)(long, long)>(GetProcAddress(m_h_library, "McammSetBitsPerPixel"));
		assert(p_mcamm_set_bits_per_pixel);
		// Binning
		p_mcamm_get_current_binning = reinterpret_cast<long (WINAPI*)(long, long*)>(GetProcAddress(m_h_library, "McammGetCurrentBinning"));
		assert(p_mcamm_get_current_binning);
		p_mcamm_has_binning = reinterpret_cast<int (WINAPI*)(long, long, BOOL*)>(GetProcAddress(m_h_library, "McammHasBinning"));
		assert(p_mcamm_has_binning);
		p_mcamm_set_binning = reinterpret_cast<long (WINAPI*)(long, long)>(GetProcAddress(m_h_library, "McammSetBinning"));
		assert(p_mcamm_set_binning);
		// Region Of Interest
		p_mcamm_get_currentframe_size = reinterpret_cast<long (WINAPI*)(long, RECT*)>(GetProcAddress(m_h_library, "McammGetCurrentFrameSize"));
		assert(p_mcamm_get_currentframe_size);
		p_mcamm_has_frame = reinterpret_cast<int (WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammHasFrame"));
		assert(p_mcamm_has_frame);
		p_mcamm_set_frame_size = reinterpret_cast<long (WINAPI*)(long, RECT*)>(GetProcAddress(m_h_library, "McammSetFrameSize"));
		assert(p_mcamm_set_frame_size);
		// Exposure Time
		p_mcamm_get_current_exposure = reinterpret_cast<long (WINAPI*)(long, long*)>(GetProcAddress(m_h_library, "McammGetCurrentExposure"));
		assert(p_mcamm_get_current_exposure);
		p_mcammget_exposure_range = reinterpret_cast<long (WINAPI*)(long, long*, long*, long*)>(GetProcAddress(m_h_library, "McammGetExposureRange"));
		assert(p_mcammget_exposure_range);
		p_mcamm_set_exposure = reinterpret_cast<long (WINAPI*)(long, long)>(GetProcAddress(m_h_library, "McammSetExposure"));
		assert(p_mcamm_set_exposure);
		// Acquisition
		p_mcammacquisition_ex = reinterpret_cast<long (WINAPI*)(long, unsigned short*, long, int(__cdecl*)(long, long, eMcamStatus, void*), void*)
		>(GetProcAddress(m_h_library, "McammAcquisitionEx"));
		assert(p_mcammacquisition_ex);
		p_mcamm_start_fastacquisition = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammStartFastAcquisition"));
		assert(p_mcamm_start_fastacquisition);
		p_mcamm_abort_fastacquisition = reinterpret_cast<long (WINAPI*)(long)>(GetProcAddress(m_h_library, "McammAbortFastAcquisition"));
		assert(p_mcamm_abort_fastacquisition);
		p_mcamm_is_fastacquisition_ready = reinterpret_cast<long (WINAPI*)(long, unsigned short*, long, int)>(GetProcAddress(
			m_h_library, "McammIsFastAcquisitionReady"));
		assert(p_mcamm_is_fastacquisition_ready);
		p_mcamm_get_current_data_size = reinterpret_cast<long (WINAPI*)(long, long*, long*)>(GetProcAddress(m_h_library, "McammGetCurrentDataSize"));
		assert(p_mcamm_get_current_data_size);
		// White Balance
		p_mcamm_get_current_white_balance = reinterpret_cast<long (WINAPI*)(long, double*, double*, double*)>(GetProcAddress(
			m_h_library, "McammGetCurrentWhiteBalance"));
		assert(p_mcamm_get_current_white_balance);
		p_mcamm_set_image_white_balance = reinterpret_cast<long (WINAPI*)(long, double, double, double)>(GetProcAddress(
			m_h_library, "McammSetImageWhiteBalance"));
		assert(p_mcamm_set_image_white_balance);
		p_mcamm_set_white_balance = reinterpret_cast<long (WINAPI*)(long, double, double, double)>(GetProcAddress(m_h_library, "McammSetWhiteBalance"));
		assert(p_mcamm_set_white_balance);
		// White Reference
		p_mcamm_calculate_white_ref_ex = reinterpret_cast<long (WINAPI*)(long, int(__cdecl*)(long, long, eMcamStatus, void*), void*)>(GetProcAddress(
			m_h_library, "McammCalculateWhiteRefEx"));
		assert(p_mcamm_calculate_white_ref_ex);
		p_mcamm_get_current_white_ref = reinterpret_cast<BOOL(WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammGetCurrentWhiteRef"));
		assert(p_mcamm_get_current_white_ref);
		p_mcamm_has_white_ref = reinterpret_cast<BOOL(WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammHasWhiteRef"));
		assert(p_mcamm_has_white_ref);
		p_mcamm_set_white_ref = reinterpret_cast<long (WINAPI*)(long, BOOL)>(GetProcAddress(m_h_library, "McammSetWhiteRef"));
		assert(p_mcamm_set_white_ref);
		p_mcamm_get_white_table = reinterpret_cast<long (WINAPI*)(long, short*)>(GetProcAddress(m_h_library, "McammGetWhiteTable"));
		assert(p_mcamm_get_white_table);
		p_mcamm_set_white_table = reinterpret_cast<long (WINAPI*)(long, short*)>(GetProcAddress(m_h_library, "McammSetWhiteTable"));
		assert(p_mcamm_set_white_table);
		// Black Reference
		p_mcamm_calculate_black_ref_ex = reinterpret_cast<long (WINAPI*)(long, int(__cdecl*)(long, long, eMcamStatus, void*), void*)>(GetProcAddress(
			m_h_library, "McammCalculateBlackRefEx"));
		assert(p_mcamm_calculate_black_ref_ex);
		p_mcamm_get_current_black_ref = reinterpret_cast<BOOL(WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammGetCurrentBlackRef"));
		assert(p_mcamm_get_current_black_ref);
		p_mcamm_has_black_ref = reinterpret_cast<BOOL(WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammHasBlackRef"));
		assert(p_mcamm_has_black_ref);
		p_mcamm_set_black_ref = reinterpret_cast<long (WINAPI*)(long, BOOL)>(GetProcAddress(m_h_library, "McammSetBlackRef"));
		assert(p_mcamm_set_black_ref);
		p_mcamm_restore_black_ref = reinterpret_cast<long (WINAPI*)(long, unsigned short*, long)>(GetProcAddress(m_h_library, "McammRestoreBlackRef"));
		assert(p_mcamm_restore_black_ref);
		p_mcamm_save_black_ref = reinterpret_cast<long (WINAPI*)(long, unsigned short*, long)>(GetProcAddress(m_h_library, "McammSaveBlackRef"));
		assert(p_mcamm_save_black_ref);
		// Filter Operations
		p_mcamm_is_color_processing_enabled = reinterpret_cast<BOOL(WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammIsColorProcessingEnabled"));
		assert(p_mcamm_is_color_processing_enabled);
		p_mcamm_enable_color_processing = reinterpret_cast<void (WINAPI*)(long, BOOL)>(GetProcAddress(m_h_library, "McammEnableColorProcessing"));
		assert(p_mcamm_enable_color_processing);
		p_mcamm_set_blooming_voltage = reinterpret_cast<long (WINAPI*)(long, long)>(GetProcAddress(m_h_library, "McammSetBloomingVoltage"));
		assert(p_mcamm_set_blooming_voltage);
		p_mcamm_get_blooming_voltage = reinterpret_cast<long (WINAPI*)(long, long*)>(GetProcAddress(m_h_library, "McammGetBloomingVoltage"));
		assert(p_mcamm_get_blooming_voltage);
		// Trigger In/Out
		p_mcamm_enable_hardware_trigger = reinterpret_cast<long (WINAPI*)(long, BOOL)>(GetProcAddress(m_h_library, "McammEnableHardwareTrigger"));
		assert(p_mcamm_enable_hardware_trigger);
		p_mcamm_set_hardware_trigger_polarity = reinterpret_cast<long (WINAPI*)(BOOL)>(GetProcAddress(m_h_library, "McamSetHardwareTriggerPolarity"));
		assert(p_mcamm_set_hardware_trigger_polarity);
		p_mcamm_is_hardware_trigger_enabled = reinterpret_cast<long (WINAPI*)(long, BOOL*)>(GetProcAddress(m_h_library, "McammIsHardwareTriggerEnabled"));
		assert(p_mcamm_is_hardware_trigger_enabled);
		p_mcamm_get_current_shutter_delay = reinterpret_cast<long (WINAPI*)(long, long*, long*)>(GetProcAddress(m_h_library, "McammGetCurrentShutterDelay"));
		assert(p_mcamm_get_current_shutter_delay);
		p_mcamm_set_shutter_delay = reinterpret_cast<long (WINAPI*)(long, long, long)>(GetProcAddress(m_h_library, "McammSetShutterDelay"));
		assert(p_mcamm_set_shutter_delay);
		p_mcamm_shutter_control = reinterpret_cast<long (WINAPI*)(long, BOOL, BOOL)>(GetProcAddress(m_h_library, "McammShutterControl"));
		assert(p_mcamm_shutter_control);
		p_close_mr3 = reinterpret_cast<long(WINAPI*)()>(GetProcAddress(m_h_library, "McamClose"));
		assert(p_close_mr3);
		p_init_mr3 = reinterpret_cast<long(WINAPI*)()>(GetProcAddress(m_h_library, "McamInit"));
		assert(p_init_mr3);
		p_mcamm_get_current_readout_time = reinterpret_cast<long (WINAPI*)(long, long*)>(GetProcAddress(m_h_library, "McammGetCurrentReadouttime"));
		assert(p_mcamm_get_current_readout_time);
	}
	long cam;

	long mcamm_nextacquisition(unsigned short*& p_image_data,
		const long allocated_size, const long microseconds) const
	{
		return p_mcamm_nextacquisition(cam, p_image_data, allocated_size, microseconds);
	}
	bool acquiring;
	long mcamm_initializeacquisition()
	{
		const auto ret = !acquiring ? p_mcamm_initializeacquisition(cam) : NOERR;
		acquiring = true;
		return ret;
	}
	long mcamm_finalizeacquisition()
	{
		acquiring = false;
		const auto ret = acquiring ? p_mcamm_finalizeacquisition(cam) : NOERR;
		return ret;
	}

	long mcamm_get_current_readout_time(long* time)
	{
		return p_mcamm_get_current_readout_time(cam, time);
	}


	void mcamm_get_current_resolution(long* idx) const
	{
		idx[0] = p_mcamm_get_current_resolution(cam);
	}
	long mcamm_get_current_frame_size(RECT* p_rect) const
	{
		return p_mcamm_get_currentframe_size(cam, p_rect);
	}
	long mcamm_get_current_exposure(long* microseconds) const
	{
		return p_mcamm_get_current_exposure(cam, microseconds);
	}
	long mcammget_exposure_range(long* p_min, long* p_max, long* p_inc) const
	{
		return p_mcammget_exposure_range(cam, p_min, p_max, p_inc);
	}
	long mcamm_get_blooming_voltage(long*& volt) const
	{
		return p_mcamm_get_blooming_voltage(cam, volt);
	}
	long mcamm_get_current_data_size(long* p_width, long* p_height) const
	{
		return p_mcamm_get_current_data_size(cam, p_width, p_height);
	}
	long mcamm_get_current_bits_per_pixel() const
	{
		return p_mcamm_get_current_bits_per_pixel(cam);
	}
	BOOL mcamm_set_frame_size(RECT* p_rect) const
	{
		return p_mcamm_set_frame_size(cam, p_rect);
	}
	zeiss_camera_imp() : p_init_mr3{ nullptr }, acquiring(false)  // NOLINT(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
	{
		m_h_library = LoadLibraryA("mrfw64.dll");
		if (!m_h_library)
		{
			qli_runtime_error("Something terrible with the zeiss library");
		}
		const auto p_num_cams_mr3 = reinterpret_cast<long(WINAPI*)()>(GetProcAddress(
			m_h_library, "McamGetNumberofCameras"));
		const auto cameras = p_num_cams_mr3();
		if (cameras < 1)
		{
			qli_runtime_error("Is your camera running, you better go catch it!");
		}
		cam = 0;
		load_functions();
		AXIO_SAFE_CALL(p_init_mr3());
	}

	~zeiss_camera_imp()
	{
		AXIO_SAFE_CALL(p_close_mr3());
		FreeLibrary(m_h_library);
	}
};


zeiss_camera::zeiss_camera(const int camera_idx, QObject* parent)
	: camera_device(camera_device_features(false, false, true, false, camera_contrast_features(camera_chroma::monochrome, demosaic_mode::no_processing, { 30,4096 })), camera_idx, parent), f_(nullptr), camera_buffer_queue_m_kill_(false)
{
	f_ = std::make_unique< zeiss_camera_imp>();
	bin_modes.emplace_back(1);
	//Hard coding doesn't matter too much because nobody uses this camera.
	auto five_twelve = camera_aoi(512, 512, 0, 0);
	five_twelve.re_center_and_fixup(1388, 1040);
	aois.push_back(five_twelve);
	aois.emplace_back(1388, 1040, 0, 0);
	common_post_constructor();
}

zeiss_camera::~zeiss_camera() = default;

void zeiss_camera::trigger_internal()
{
	//captured one, store it in the buffer, now 'inside the camera'
	unsigned short* ptr_in;
	{
		std::unique_lock<std::mutex> lk(free_buffer_m_);
		const auto predicate = [&] {return camera_buffer_queue_m_kill_ || !free_buffer_.empty(); };
		camera_buffer_queue_cv_.wait(lk, predicate);
		if (camera_buffer_queue_m_kill_)
		{
			return;
		}
		ptr_in = free_buffer_.front();
		free_buffer_.pop();
	}
	const auto bytes = get_sensor_bytes(camera_configuration_);
	const auto microseconds_us = exposure_.count();//exposure time for *NEXT* Image, oops
	const auto status = f_->mcamm_nextacquisition(ptr_in, bytes, microseconds_us);
	AXIO_SAFE_CALL(status);//this is critical and kills the program so we don't much care about leaks
	{
		{
			std::unique_lock<std::mutex> lk(inside_camera_m_);
			inside_camera_.push(ptr_in);
		}
		inside_camera_cv_.notify_one();
	}
}

bool zeiss_camera::capture_internal(const camera_frame_processing_function & fill_me, const frame_meta_data_before_acquire & meta_data, const std::chrono::microseconds & timeout)
{
	//time_slice ts("Capture");
	unsigned short* ptr_out = nullptr;
	{
		std::unique_lock<std::mutex> lk(inside_camera_m_);
		auto predicate = [&] {return !inside_camera_.empty(); };
		const auto success = inside_camera_cv_.wait_for(lk, timeout, predicate);
		if (success)
		{
			ptr_out = inside_camera_.front();
			inside_camera_.pop();
		}
	}
	const auto success = ptr_out != nullptr;
	if (success)
	{
		const auto frame_size = aois.at(camera_configuration_.aoi_index).to_frame_size();
		const auto meta_data_after = frame_meta_data(meta_data, timestamp());
		const auto samples_per_pixel = is_forced_color() ? 3 : 1;
		const auto info = image_info(frame_size, samples_per_pixel, image_info::complex::no);
		const auto frame = camera_frame<unsigned short>(ptr_out, info, meta_data_after);
#if 0
		{
			{
				static auto testery_doop = 0;
				testery_doop = testery_doop + 1;
				if (testery_doop > 10)
				{
					const auto item = get_sensor_size(camera_configuration_);
					auto logged = std::to_string(testery_doop) + "_place_2.tif";
					write_tif(logged, ptr_out, item.width, item.height, 1);
					const auto here = 0;
				}
			}
		}
#endif
		fill_me(frame);
		{
			std::unique_lock<std::mutex> lk(free_buffer_m_);
			free_buffer_.push(ptr_out);
			camera_buffer_queue_cv_.notify_one();
		}
	}
	return success;
}

void zeiss_camera::fix_camera_internal()
{
	//not implemented good luck :-)
}

void zeiss_camera::apply_settings_internal(const camera_config & new_config)
{
	const auto aoi = aois.at(new_config.aoi_index);
	//right, bottom
	RECT rectangle = {
		static_cast<LONG>(aoi.left),
		static_cast<LONG>(aoi.top),
		static_cast<LONG>(aoi.width + aoi.left),
		static_cast<LONG>(aoi.height + aoi.top) };
	AXIO_SAFE_CALL(f_->mcamm_set_frame_size(&rectangle));
	//
	const auto bits_per_pixel = f_->mcamm_get_current_bits_per_pixel();
	const auto canonical_size = 12;
	if (bits_per_pixel % canonical_size != 0)
	{
		qli_runtime_error("Unsupported size");
	}
	const auto configured_sample_per_pixel = bits_per_pixel / canonical_size;
	chroma = configured_sample_per_pixel == 3 ? camera_chroma::forced_color : camera_chroma::monochrome;
	//
	inside_camera_ = std::queue<unsigned short*>();
	free_buffer_ = inside_camera_;
	camera_buffer_.resize(internal_buffers);
	long columns, rows;
	AXIO_SAFE_CALL(f_->mcamm_get_current_data_size(&columns, &rows));
	for (auto&& item : camera_buffer_)
	{
		item.resize(columns * rows * configured_sample_per_pixel);
		auto* ptr = item.data();
		free_buffer_.push(ptr);
	}
}

void zeiss_camera::set_cooling_internal(bool)
{
	std::cout << "set cooling internal time internal in source file" << std::endl;
	// qli_not_implemented();
}

void zeiss_camera::set_exposure_internal(const std::chrono::microseconds&)
{
	//so setting the exposure should skip a bunch of frames? until we find the one with the correct exposure
}

void zeiss_camera::print_debug(std::ostream&)
{

}

void zeiss_camera::start_capture_internal()
{
	AXIO_SAFE_CALL(f_->mcamm_initializeacquisition());
	//maybe clear old queue
	for (size_t i = 0; i < inside_camera_.size(); i++)
	{
		auto* ptr = inside_camera_.front();
		inside_camera_.pop();
		free_buffer_.push(ptr);
	}
}

void zeiss_camera::stop_capture_internal()
{
	AXIO_SAFE_CALL(f_->mcamm_finalizeacquisition());
}

std::chrono::microseconds zeiss_camera::get_min_exposure_internal()
{
	static long exp_min = 0, exp_max = 0, exp_inc = 0;
	//doesn't change for now
	if (exp_min == 0)
	{
		AXIO_SAFE_CALL(f_->mcammget_exposure_range(&exp_min, &exp_max, &exp_inc));
	}
	return std::chrono::microseconds(exp_min);
}

std::chrono::microseconds zeiss_camera::get_readout_time_internal()
{
	static long time = 0;
	AXIO_SAFE_CALL(f_->mcamm_get_current_readout_time(&time));
	return std::chrono::microseconds(time);
	// return std::chrono::microseconds(300);
}

// std::chrono::microseconds zeiss_camera::get_min_cycle_internal()
// {
// 	const auto max_fps = 13;
// 	const auto time_long = static_cast<long>(ceil(1000 / (1. * max_fps)) * 1000);
// 	return std::chrono::microseconds(time_long);//guess
// }

#endif