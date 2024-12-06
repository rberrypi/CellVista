#include "stdafx.h"
#include "hamamatsu_camera.h"
#if CAMERA_PRESENT_HAMAMATSU == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include <iostream>
#include "time_slice.h"
#include <dcamapi4.h>
#include <dcamprop.h>
#include <queue>
#include "qli_runtime_error.h"
#include <sstream>
#pragma comment(lib, "dcamapi.lib")

#define DCAM_API_SAFE_CALL(err) dcam_safe_call(err,__FILE__,__LINE__,true)
#define DCAM_GET_ERROR() dcam_safe_call(false,__FILE__,__LINE__,false)
inline void dcam_safe_call(const DCAMERR code, const char* file, const int line, const bool throws)
{
	if (failed(code))
	{
		std::stringstream ss;
		ss << "Hamamatsu Error " << file << " : " << line << std::endl;
		if (throws)
		{
			qli_runtime_error(ss.str());
		}
	}
}

HDCAM	hdcam;
DCAMWAIT_OPEN	m_hwait;
hamamatsu_device::hamamatsu_device(const int camera_idx, QObject* parent) : camera_device(camera_device_features(true, false, false, false, camera_contrast_features(camera_chroma::monochrome, demosaic_mode::no_processing, { 70,65535 })), camera_idx, parent)
{
	{
		DCAMAPI_INIT paraminit{};
		std::memset(&paraminit, 0, sizeof paraminit);
		paraminit.size = sizeof paraminit;
		DCAM_API_SAFE_CALL(dcamapi_init(&paraminit));
	}
	{
		DCAMDEV_OPEN param{};
		std::memset(&param, 0, sizeof(param));
		param.size = sizeof param;
		param.index = 0;
		DCAM_API_SAFE_CALL(dcamdev_open(&param));
		hdcam = param.hdcam;
	}
	{
		memset(&m_hwait, 0, sizeof m_hwait);
		m_hwait.size = sizeof m_hwait;
		m_hwait.hdcam = hdcam;
		DCAM_API_SAFE_CALL(dcamwait_open(&m_hwait));
	}
	{
		//build AOIs
		DCAMPROP_ATTR width_prop{};
		memset(&width_prop, 0, sizeof(DCAMPROP_ATTR));
		width_prop.cbSize = sizeof(DCAMPROP_ATTR);
		width_prop.iProp = DCAM_IDPROP_IMAGE_WIDTH;
		DCAM_API_SAFE_CALL(dcamprop_getattr(hdcam, &width_prop));

		DCAMPROP_ATTR height_prop{};
		memset(&height_prop, 0, sizeof(DCAMPROP_ATTR));
		height_prop.cbSize = sizeof(DCAMPROP_ATTR);
		height_prop.iProp = DCAM_IDPROP_IMAGE_HEIGHT;
		DCAM_API_SAFE_CALL(dcamprop_getattr(hdcam, &height_prop));


		aois = {
			camera_aoi(width_prop.valuemax, height_prop.valuemax, 0, 0),
			camera_aoi(1920, 1072, 0, 0),
			camera_aoi(1776, 1760, 0, 0),
			camera_aoi(1440, 1080, 0, 0),
			camera_aoi(1928, 512, 0, 0),
			camera_aoi(1024, 1024, 0, 0),
			camera_aoi(768, 768, 0, 0),
			camera_aoi(512, 512, 0, 0),
			camera_aoi(256, 256, 0, 0)
		};
		for (auto& roi : aois)
		{
			roi.re_center_and_fixup(width_prop.valuemax, height_prop.valuemax);
		}
		bin_modes.emplace_back(1);
		bin_modes.emplace_back(2);
		bin_modes.emplace_back(4);
		bin_modes.emplace_back(8);

	}
	//something with cooler control here
	common_post_constructor();
}

hamamatsu_device::~hamamatsu_device()
{
	DCAM_API_SAFE_CALL(dcamdev_close(hdcam));
	DCAM_API_SAFE_CALL(dcamapi_uninit());
}

void  hamamatsu_device::trigger_internal()
{
	DCAM_API_SAFE_CALL(dcamcap_firetrigger(hdcam));
}

bool hamamatsu_device::capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds&)
{
	{
		DCAMWAIT_START	parameter_wait{};
		memset(&parameter_wait, 0, sizeof(parameter_wait));
		parameter_wait.size = sizeof(parameter_wait);
		parameter_wait.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY;
		parameter_wait.timeout = DCAMWAIT_TIMEOUT_INFINITE;
		const auto err = dcamwait_start(m_hwait.hwait, &parameter_wait);
		if (failed(err))
		{
			return false;
		}
	}
	{
		DCAMBUF_FRAME frame{};
		memset(&frame, 0, sizeof(frame));
		frame.size = sizeof(frame);
		frame.iFrame = -1;		// latest frame
		DCAM_API_SAFE_CALL(dcambuf_lockframe(hdcam, &frame));
		//
		//todo some size verification goes here
		//
		const std::chrono::microseconds microseconds(frame.timestamp.microsec);
		const std::chrono::seconds seconds(frame.timestamp.sec);
		const auto alternative_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(seconds) + microseconds;
		const frame_size size(frame.width, frame.height);
		const image_info info(size, 1, image_info::complex::no);
		const frame_meta_data meta_data_after(meta_data, alternative_timestamp);
		const camera_frame<unsigned short> fill_you(static_cast<unsigned short*>(frame.buf), info, meta_data_after);
		fill_me(fill_you);

	}
	return true;
}

void hamamatsu_device::fix_camera_internal()
{
	//not sure how to fix this camera, maybe stop everything? and try again?
}

void hamamatsu_device::apply_settings_internal(const camera_config& new_config)
{
	{
		const auto prop = new_config.mode == camera_mode::burst ? DCAMPROP_TRIGGERSOURCE__INTERNAL:  DCAMPROP_TRIGGERSOURCE__SOFTWARE ;
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_TRIGGERSOURCE, prop));
	}
	//set resolution & binning here
	const auto set_aoi = [&](const camera_aoi& aoi) {
		const std::array<_DCAMPROPMODEVALUE, 4> bin_modes = { DCAMPROP_BINNING__1,DCAMPROP_BINNING__2,DCAMPROP_BINNING__4,DCAMPROP_BINNING__8 };
		const auto bin_mode = bin_modes.at(new_config.bin_index);
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__OFF));
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBARRAYHSIZE, aoi.width));
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBARRAYHPOS, aoi.left));
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBARRAYVSIZE, aoi.height));
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBARRAYVPOS, aoi.top));
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_BINNING, bin_mode));
		DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__ON));
	};
	//
	//
	DCAM_API_SAFE_CALL(dcambuf_release(hdcam));
	const auto& aoi = aois.at(new_config.aoi_index);
	set_aoi(aoi);
	DCAM_API_SAFE_CALL(dcambuf_alloc(hdcam, internal_frame_count));
}

void hamamatsu_device::set_exposure_internal(const std::chrono::microseconds& exposure)
{
	//todo will die if you set this number too small, for example you can't have a zero exposure, right?tore

	// you can set this whenever?
	const auto exposure_time_seconds_chrono = std::chrono::duration<double>(exposure);
	const auto exposure_time_seconds = exposure_time_seconds_chrono.count();
	DCAM_API_SAFE_CALL(dcamprop_setvalue(hdcam, DCAM_IDPROP_EXPOSURETIME, exposure_time_seconds));
#if _DEBUG
	double value;
	DCAM_API_SAFE_CALL(dcamprop_getvalue(hdcam, DCAM_IDPROP_EXPOSURETIME, &value));
#endif
}

void hamamatsu_device::print_debug(std::ostream&)
{
	//
}

void hamamatsu_device::start_capture_internal()
{
	dcamcap_start(hdcam, DCAMCAP_START_SEQUENCE);
}

void hamamatsu_device::stop_capture_internal()
{
	//if this fails, then you done goofed, and we should probably fix that bug
	DCAM_API_SAFE_CALL(dcamcap_stop(hdcam));
}

std::chrono::microseconds hamamatsu_device::get_min_exposure_internal()
{
	//see DCAM_IDPROP_TIMING_INVALIDEXPOSUREPEROID
	//todo write something here!
	return ms_to_chrono(1);
}

std::chrono::microseconds hamamatsu_device::get_readout_time_internal()
{
	double blanking_timing = 0;
	const auto prop = DCAM_IDPROP_TIMING_READOUTTIME;
	DCAM_API_SAFE_CALL(dcamprop_getvalue(hdcam, prop, &blanking_timing));
	const auto exposure_time_seconds_chrono = std::chrono::duration<double>(blanking_timing);
	const auto exposure_time_seconds_chrono_micro = std::chrono::duration_cast<std::chrono::microseconds>(exposure_time_seconds_chrono);
	return exposure_time_seconds_chrono_micro;
}

std::chrono::microseconds hamamatsu_device::get_transfer_time_internal()
{
	//todo this might be wrong because it can overflow?
	double blanking_timing = 0;
	const auto prop = DCAM_IDPROP_TIMING_MINTRIGGERINTERVAL;
	DCAM_API_SAFE_CALL(dcamprop_getvalue(hdcam, prop, &blanking_timing));
	const auto exposure_time_seconds_chrono = std::chrono::duration<double>(blanking_timing);
	const auto exposure_time_seconds_chrono_micro = std::chrono::duration_cast<std::chrono::microseconds>(exposure_time_seconds_chrono);
	return exposure_time_seconds_chrono_micro;
}

/*
 * 	double blanking_timing = 0;
	const auto prop = DCAM_IDPROP_TIMING_MINTRIGGERINTERVAL;
	DCAM_API_SAFE_CALL(dcamprop_getvalue(hdcam, prop, &blanking_timing));
	const auto exposure_time_seconds_chrono = std::chrono::duration<double>(blanking_timing);
	const auto exposure_time_seconds_chrono_micro = std::chrono::duration_cast<std::chrono::microseconds>(exposure_time_seconds_chrono);
	return exposure_time_seconds_chrono_micro;
 */

void hamamatsu_device::set_cooling_internal(bool)
{
	qli_not_implemented();
}

void hamamatsu_device::flush_camera_internal_buffer()
{
	//needs to be implemented (probably?)
	if (camera_configuration_ != camera_config::invalid_cam_config()) //dirty hack
	{
		apply_settings_internal(camera_configuration_);//double free, maybe in some cases
	}
}

bool hamamatsu_device::capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& exposure, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& process_function)
{
	DCAM_API_SAFE_CALL(dcamcap_start(hdcam, DCAMCAP_START_SEQUENCE));
	for (auto it = frames.first; it < frames.second; ++it)
	{
		{
			DCAMWAIT_START	parameter_wait{};
			memset(&parameter_wait, 0, sizeof(parameter_wait));
			parameter_wait.size = sizeof(parameter_wait);
			parameter_wait.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY;
			parameter_wait.timeout = DCAMWAIT_TIMEOUT_INFINITE;
			const auto err = dcamwait_start(m_hwait.hwait, &parameter_wait);
			if (failed(err))
			{
				return false;
			}
		}
		{
			DCAMBUF_FRAME frame{};
			memset(&frame, 0, sizeof(frame));
			frame.size = sizeof(frame);
			frame.iFrame = -1;		// latest frame
			DCAM_API_SAFE_CALL(dcambuf_lockframe(hdcam, &frame));
			//
			//todo some size verification goes here
			//
			const std::chrono::microseconds microseconds(frame.timestamp.microsec);
			const std::chrono::seconds seconds(frame.timestamp.sec);
			const auto alternative_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(seconds) + microseconds;
			const frame_size size(frame.width, frame.height);
			const image_info info(size, 1, image_info::complex::no);
			const frame_meta_data meta_data_after(meta_data, alternative_timestamp);
			const camera_frame<unsigned short> fill_you(static_cast<unsigned short*>(frame.buf), info, meta_data_after);
			process_function(fill_you);

		}
	}
	DCAM_API_SAFE_CALL(dcamcap_stop( hdcam ));
	return true;
}

#endif