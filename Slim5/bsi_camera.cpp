#include "stdafx.h"
#if CAMERA_PRESENT_BSI==CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#pragma comment(lib, "pvcam64.lib") 
#include <iostream>
#include <sstream>
#include "time_slice.h"
#include "bsi_camera.h"
#include "bsi_camera_common.h"
#include "qli_runtime_error.h"
#define BSI_SAFE_CALL(err) bsi_safe_call(err,__FILE__,__LINE__)
inline void bsi_safe_call(const rs_bool err, const char* file, const int line)
{
	if (PV_OK != err)
	{
		char msg[200] = { 0 };
		const auto code = pl_error_code();
		pl_error_message(code, msg);
		std::stringstream ss;
		ss << "PV Failure Code: " << msg << " @" << line << ":" << file << ": ";
		CloseCameraAndUninit();
		qli_runtime_error(ss.str());
	}
}

bsi_device::bsi_device(const int camera_idx, QObject* parent) :camera_device(camera_device_features(false, false, false, false, camera_contrast_features(camera_chroma::monochrome, demosaic_mode::no_processing, { 0, 2048 })), camera_idx, parent), last_exposure(std::numeric_limits<unsigned int>::max())
{
	{
		uns16 pvcam_version;
		BSI_SAFE_CALL(pl_pvcam_get_ver(&pvcam_version));
		std::cout << "PV Version " << pvcam_version << std::endl;
	}
	BSI_SAFE_CALL(InitAndOpenFirstCamera() ? PV_OK : PV_FAIL);
	{

	}
	bin_modes.emplace_back(camera_bin(1));
	bin_modes.emplace_back(camera_bin(2));
	bin_modes.emplace_back(camera_bin(4));
	const auto divisible_by_four = [](const auto x) {return 4 * floor(x / 4); };
	aois.emplace_back(camera_aoi(divisible_by_four(g_SensorResX * 1), divisible_by_four(g_SensorResY * 1), 0, 0));
	for (const auto value : { 0.75,0.5,0.25,0.125 })
	{
		aois.emplace_back(camera_aoi(divisible_by_four(g_SensorResX * value), divisible_by_four(g_SensorResY * value), 0, 0));
		aois.emplace_back(camera_aoi(divisible_by_four(g_SensorResX), divisible_by_four(g_SensorResY * value), 0, 0));
	}
	for (auto& aoi : aois)
	{
		aoi.re_center_and_fixup(g_SensorResX, g_SensorResY, 2);
	}
	common_post_constructor();
}

bsi_device:: ~bsi_device()
{
	CloseCameraAndUninit();
}

void bsi_device::trigger_internal()
{
	//nothing to do, maybe just some queue to pretend we are actually doing something?
}


void set_PARAM_TYPE_int16(const uns16 param, const int16 value)
{
	rs_bool avail;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_AVAIL, &avail));
	if (!avail)
	{
		std::cout << "Warning Not Available" << std::endl;
		return;
	}
	uns16 access;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_ACCESS, &access));
	if (access != ACC_READ_WRITE)
	{
		qli_runtime_error("No Access");
	}
	//
	uns16 param_type;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_TYPE, &param_type));
	if (param_type != TYPE_INT16)
	{
		qli_not_implemented();		
	}
	uns32 count;
	auto current = value, min = value, max = value, after_set = value;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_COUNT, &count));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &current));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MIN, &min));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MAX, &max));
	BSI_SAFE_CALL(pl_set_param(g_hCam, param, (void*)&value));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &after_set));
	std::cout << "Setting " << param << " , " << value << " [" << static_cast<int>(min) << "," << static_cast<int>(max) << "]" << std::endl;
	if (after_set != value)
	{
		qli_runtime_error("Something Wrong");
	}
}

void set_PARAM_TYPE_uns16(const uns16 param, const uns16 value)
{
	rs_bool avail;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_AVAIL, &avail));
	if (!avail)
	{
		std::cout << "Warning Not Available" << std::endl;
		return;
	}
	uns16 access;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_ACCESS, &access));
	if (access != ACC_READ_WRITE)
	{
		qli_runtime_error("No Access");
	}
	//
	uns16 param_type;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_TYPE, &param_type));
	if (param_type != TYPE_UNS16)
	{
		qli_not_implemented();
	}
	// This part changes
	uns32 count;
	auto current = value, min = value, max = value, after_set = value;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_COUNT, &count));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &current));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MIN, &min));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MAX, &max));
	BSI_SAFE_CALL(pl_set_param(g_hCam, param, (void*)&value));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &after_set));
	std::cout << "Setting " << param << " , " << value << " [" << static_cast<int>(min) << "," << static_cast<int>(max) << "]" << std::endl;
	if (after_set != value)
	{
		qli_runtime_error();
	}
}

void set_PARAM_TYPE_uns8(const uns32 param, const uns8 value)
{
	rs_bool avail;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_AVAIL, &avail));
	if (!avail)
	{
		std::cout << "Warning Not Available" << std::endl;
		return;
	}
	uns16 access;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_ACCESS, &access));
	if (access != ACC_READ_WRITE)
	{
		qli_runtime_error("No Access");
	}
	//
	uns16 param_type;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_TYPE, &param_type));
	if (param_type != TYPE_UNS8)
	{
		qli_not_implemented();
	}
	// This part changes
	uns32 count;
	auto current = value, min = value, max = value, after_set = value;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_COUNT, &count));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &current));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MIN, &min));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MAX, &max));
	BSI_SAFE_CALL(pl_set_param(g_hCam, param, (void*)&value));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &after_set));
	std::cout << "Setting " << param << " , " << static_cast<int>(value) << " [" << static_cast<int>(min) << "," << static_cast<int>(max) << "]" << std::endl;
	if (after_set != value)
	{
		qli_runtime_error();
	}
}

void set_PARAM_TYPE_enum(const uns32 param, const int32 value)
{
	rs_bool avail;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_AVAIL, &avail));
	if (!avail)
	{
		std::cout << "Warning Not Available" << std::endl;
		return;
	}
	uns16 access;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_ACCESS, &access));
	if (access != ACC_READ_WRITE)
	{
		qli_runtime_error("No Access");
	}
	//
	uns16 param_type;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_TYPE, &param_type));
	if (param_type != TYPE_ENUM)
	{
		qli_not_implemented();
	}
	// This part changes
	uns32 count;
	auto current = value, min = value, max = value, after_set = value;
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_COUNT, &count));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &current));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MIN, &min));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_MAX, &max));
	BSI_SAFE_CALL(pl_set_param(g_hCam, param, (void*)&value));
	BSI_SAFE_CALL(pl_get_param(g_hCam, param, ATTR_CURRENT, &after_set));
	const auto text_buffer_length = 256;
	static char text_buffero[text_buffer_length];
	std::cout << "Enum Variants" << std::endl;
	for (auto i = 0; i < count; ++i)
	{
		int32 value_as_enum;
		BSI_SAFE_CALL(pl_get_enum_param(g_hCam, param, i, &value_as_enum, text_buffero, text_buffer_length));
		std::cout << "Values : " << value_as_enum << ":" << text_buffero << std::endl;
	}
	std::cout << "Setting " << param << " , " << value << " [" << static_cast<int>(min) << "," << static_cast<int>(max) << "]" << std::endl;
	if (after_set != value)
	{
		qli_runtime_error();
	}
}

void set_PARAM_READOUT_PORT(int32 value)
{
	const auto param = PARAM_READOUT_PORT;
	BSI_SAFE_CALL(pl_set_param(g_hCam, param, (void*)&value));
}

void set_PARAM_SPDTAB_INDEX(int16 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_SPDTAB_INDEX, (void*)&item));
}

void set_PARAM_GAIN_INDEX(int16 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_GAIN_INDEX, (void*)&item));
}

void set_PARAM_CLEAR_CYCLES(uns16 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_CLEAR_CYCLES, (void*)&item));
}

void set_PARAM_CLEAR_MODE(int32 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_CLEAR_MODE, (void*)&item));
}

void set_PARAM_PMODE(int32 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_PMODE, (void*)&item));
}

void set_PARAM_METADATA_ENABLED(rs_bool item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_METADATA_ENABLED, (void*)&item));
}

void set_PARAM_TRIGTAB_SIGNAL(int32 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_TRIGTAB_SIGNAL, (void*)&item));
}

void set_PARAM_LAST_MUXED_SIGNAL(uns8 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_LAST_MUXED_SIGNAL, (void*)&item));
}

void set_PARAM_EXP_RES_INDEX(uns16 item)
{
	BSI_SAFE_CALL(pl_set_param(g_hCam, PARAM_EXP_RES_INDEX, (void*)&item));
}

bool bsi_device::capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout)
{
	//time_slice ts("BIS Capture Internal");
	 //do a whole acquisition here (?)
	const uns32 exposure_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(meta_data.exposure_time).count();
	const auto aoi = this->aois.at(camera_configuration_.aoi_index);
	const auto bin_mode = this->bin_modes.at(camera_configuration_.bin_index);
	auto configured_region = [&]
	{
		// The sensor region width is then calculated as <tt>s2 - s1 + 1</tt>.
		// The resulting image width would be <tt>(s2 - s1 + 1) / sbin< / tt>.
		rgn_type required_region;
		required_region.s1 = aoi.left;
		required_region.s2 = aoi.left + aoi.width - 1;
		required_region.sbin = bin_mode.s;
		required_region.p1 = aoi.top;
		required_region.p2 = aoi.top + aoi.height - 1;
		required_region.pbin = bin_mode.s;
		return required_region;
	}();
	static auto last_camera_config = camera_config::invalid_cam_config();
	if (exposure_time_ms != last_exposure || camera_configuration_ != last_camera_config)
	{

		if (0)
		{
			set_PARAM_TYPE_enum(PARAM_READOUT_PORT, 0);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 1);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 2);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 3);
			set_PARAM_TYPE_int16(PARAM_SPDTAB_INDEX, 1);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 1);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 2);
			//

			set_PARAM_TYPE_enum(PARAM_READOUT_PORT, 0);
			set_PARAM_TYPE_int16(PARAM_SPDTAB_INDEX, 1);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 1);
			set_PARAM_TYPE_uns16(PARAM_EXP_RES_INDEX, 0);
			set_PARAM_TYPE_uns16(PARAM_EXP_RES_INDEX, 1);
			set_PARAM_TYPE_uns16(PARAM_EXP_RES_INDEX, 0);
			//
			set_PARAM_TYPE_enum(PARAM_READOUT_PORT, 0);
			set_PARAM_TYPE_int16(PARAM_SPDTAB_INDEX, 0);
			//
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 1);
			set_PARAM_TYPE_uns16(PARAM_CLEAR_CYCLES, 1);
			//
			set_PARAM_TYPE_uns16(PARAM_CLEAR_CYCLES, 1);
			set_PARAM_TYPE_enum(PARAM_CLEAR_MODE, 1);
			set_PARAM_TYPE_enum(PARAM_PMODE, 0);
			//set_PARAM_METADATA_ENABLED(true);
			set_PARAM_TYPE_enum(PARAM_TRIGTAB_SIGNAL, 0);
			set_PARAM_TYPE_uns8(PARAM_LAST_MUXED_SIGNAL, 1);
			set_PARAM_TYPE_uns16(PARAM_EXP_RES_INDEX, 0);
			//
			set_PARAM_EXP_RES_INDEX(1);
			auto volatile whats = 0;
		}

		{
			//print_debug(std::cout);
			/*
			set_PARAM_TYPE_enum(PARAM_READOUT_PORT, 0);
			set_PARAM_TYPE_int16(PARAM_SPDTAB_INDEX, 1);//100 mhz
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, 1);
			set_PARAM_TYPE_uns16(PARAM_CLEAR_CYCLES, 1);
			set_PARAM_TYPE_enum(PARAM_CLEAR_MODE, 1);
			set_PARAM_TYPE_enum(PARAM_PMODE, 0);
			set_PARAM_TYPE_enum(PARAM_TRIGTAB_SIGNAL, 0);
			set_PARAM_TYPE_uns8(PARAM_LAST_MUXED_SIGNAL, 1);
			set_PARAM_TYPE_uns16(PARAM_EXP_RES_INDEX, 0);
			*/
			const auto settings = g_SpeedTable.back();
			set_PARAM_TYPE_enum(PARAM_READOUT_PORT, settings.port.value);
			set_PARAM_TYPE_int16(PARAM_SPDTAB_INDEX, settings.speedIndex);
			set_PARAM_TYPE_int16(PARAM_GAIN_INDEX, settings.gains.back());
			set_PARAM_TYPE_uns16(PARAM_CLEAR_CYCLES, 1);
			set_PARAM_TYPE_enum(PARAM_CLEAR_MODE, 1);
			set_PARAM_TYPE_enum(PARAM_PMODE, 0);
			set_PARAM_TYPE_enum(PARAM_TRIGTAB_SIGNAL, 0);
			set_PARAM_TYPE_uns8(PARAM_LAST_MUXED_SIGNAL, 1);
			set_PARAM_TYPE_uns16(PARAM_EXP_RES_INDEX, 0);
			print_debug(std::cout);
		}

		uns32 exposure_bytes;
		BSI_SAFE_CALL(pl_exp_setup_seq(g_hCam, 1, 1, &configured_region, TIMED_MODE, exposure_time_ms, &exposure_bytes));
		data.resize(exposure_bytes / sizeof(unsigned short));
		last_exposure = exposure_time_ms;
		last_camera_config = camera_configuration_;
	}
	//
	auto ptr = data.data();
	BSI_SAFE_CALL(pl_exp_start_seq(g_hCam, ptr));
	int16 status;
	uns32 byte_cnt;
	const auto polling_delay = std::chrono::milliseconds(2);
	const auto max_retries = std::chrono::seconds(10) / polling_delay;
	auto attempt = 0;
	while (pl_exp_check_status(g_hCam, &status, &byte_cnt) && status != READOUT_COMPLETE && status != READOUT_NOT_ACTIVE)
	{
		std::this_thread::sleep_for(polling_delay);
		attempt = attempt + 1;
		if (attempt > max_retries)
		{
			std::cout << "Warning BSI Lost a Frame" << std::endl;
			return false;
		}
	}
	const frame_size frame_size(aoi.width / bin_mode.s, aoi.height / bin_mode.s);
#if _DEBUG
	{
		const auto expected_byte_count = frame_size.n() * sizeof(unsigned short);
		if (expected_byte_count > byte_cnt)
		{
			qli_runtime_error();
		}
	}
#endif
	const auto time_stamp = timestamp();
	const auto meta_data_after = frame_meta_data(meta_data, time_stamp);
	const auto frame_info = image_info(frame_size,1,image_info::complex::no);
	camera_frame<unsigned short> frame(ptr, frame_info, meta_data_after);
	fill_me(frame);
	//BSI_SAFE_CALL(pl_exp_finish_seq(g_hCam, ptr, 0));
	return true;
}

void bsi_device::fix_camera_internal()
{
	//nothing to do
}

void bsi_device::apply_settings_internal(const camera_config& new_config)
{
	//nothing to do right now
}

void bsi_device::set_cooling_internal(bool enable)
{
	qli_not_implemented();
}

void bsi_device::set_exposure_internal(const std::chrono::microseconds& exposure)
{
	//nothing to do
}

void bsi_device::print_debug(std::ostream& input)
{
	// what the heck
	std::cout << "What did we set " << std::endl;
	{
		char info[MAX_GAIN_NAME_LEN] = { 0 };
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_GAIN_NAME, ATTR_CURRENT, (void*)info));
		std::cout << "PARAM_GAIN_NAME" << " " << info << std::endl;
	}
	{
		uns32 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_FWELL_CAPACITY, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_FWELL_CAPACITY" << " " << item << std::endl;
	}
	{
		uns16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_PIX_TIME, ATTR_CURRENT, (void*)&item));
		const auto as_mhz = static_cast<float>(1000) / item;
		std::cout << "PARAM_PIX_TIME" << " " << item << " ( " << as_mhz << " MHZ )" << std::endl;
	}
	{
		uns32 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_READOUT_TIME, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_READOUT_TIME" << " " << item << std::endl;
	}
	{
		int32 item = 0, value = 0;
		char desc[MAX_GAIN_NAME_LEN] = { 0 };
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_EXPOSURE_MODE, ATTR_CURRENT, (void*)&item));
		//BSI_SAFE_CALL(pl_get_enum_param(g_hCam, PARAM_EXPOSURE_MODE, item, &value, desc, MAX_GAIN_NAME_LEN));
		std::cout << "PARAM_EXPOSURE_MODE" << " " << value << std::endl;
	}
	{
		int32 item = 0, value = 0;
		char desc[MAX_GAIN_NAME_LEN] = { 0 };
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_EXPOSE_OUT_MODE, ATTR_CURRENT, (void*)&item));
		//BSI_SAFE_CALL(pl_get_enum_param(g_hCam, PARAM_EXPOSE_OUT_MODE, item, &value, desc, MAX_GAIN_NAME_LEN));
		std::cout << "PARAM_EXPOSURE_MODE" << " " << value << std::endl;
	}
	{
		int16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_BIT_DEPTH, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_BIT_DEPTH" << " " << item << std::endl;
	}
	{
		ulong64 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_EXPOSURE_TIME, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_EXPOSURE_TIME" << " " << item << std::endl;
	}
	//
	{
		uns16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_CLEAR_CYCLES, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_CLEAR_CYCLES" << " " << item << std::endl;
	};
	{
		int32 item = 0, value = 0;
		char desc[MAX_GAIN_NAME_LEN] = { 0 };
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_CLEAR_MODE, ATTR_CURRENT, (void*)&item));
		//BSI_SAFE_CALL(pl_get_enum_param(g_hCam, PARAM_CLEAR_MODE, item, &value, desc, MAX_GAIN_NAME_LEN));
		std::cout << "PARAM_CLEAR_MODE" << " " << value << std::endl;
	};
	{
		int16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_GAIN_INDEX, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_GAIN_INDEX" << " " << item << std::endl;
	};
	{
		int32 item = 0, value = 0;
		char desc[MAX_GAIN_NAME_LEN] = { 0 };
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_PMODE, ATTR_CURRENT, (void*)&item));
		//BSI_SAFE_CALL(pl_get_enum_param(g_hCam, PARAM_PMODE, item, &value, desc, MAX_GAIN_NAME_LEN));
		std::cout << "PARAM_PMODE" << " " << value << std::endl;
	};
	{
		int16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_SPDTAB_INDEX, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_SPDTAB_INDEX" << " " << item << std::endl;
	};
	{
		int32 item = 0, value = 0;
		char desc[MAX_GAIN_NAME_LEN] = { 0 };
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_TRIGTAB_SIGNAL, ATTR_CURRENT, (void*)&item));
		//BSI_SAFE_CALL(pl_get_enum_param(g_hCam, PARAM_TRIGTAB_SIGNAL, item, &value, desc, MAX_GAIN_NAME_LEN));
		std::cout << "PARAM_TRIGTAB_SIGNAL" << " " << value << std::endl;
	};
	{
		int16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_TEMP_SETPOINT, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_TEMP_SETPOINT" << " " << item << std::endl;
	};
	{
		int16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_ADC_OFFSET, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_ADC_OFFSET" << " " << item << std::endl;
	};
	{
		uns16 item = 0;
		BSI_SAFE_CALL(pl_get_param(g_hCam, PARAM_EXP_TIME, ATTR_CURRENT, (void*)&item));
		std::cout << "PARAM_EXP_TIME" << " " << item << std::endl;
	};
	std::cout << "" << std::endl;
	std::cout << "Setting Values" << std::endl;
}

void bsi_device::start_capture_internal()
{

}

void bsi_device::stop_capture_internal()
{
	//nothng to o
}

std::chrono::microseconds bsi_device::get_min_exp_internal()
{
	return std::chrono::milliseconds(1);
}

std::chrono::microseconds bsi_device::get_min_cycle_internal()
{
	return std::chrono::milliseconds(60);
}

int bsi_device::get_internal_buffer_count() const
{
	return 1;
}

QStringList bsi_device::get_gain_names_internal() const
{
	QStringList in_modes;
	in_modes << "none";
	return in_modes;
}

void bsi_device::flush_camera_internal_buffer()
{
	//nothing to do
}
#endif