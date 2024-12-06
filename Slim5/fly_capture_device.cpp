#include "stdafx.h"
#if CAMERA_PRESENT_FLYCAPTURE == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "fly_capture_device.h"
#include <FlyCapture2.h>
#include "qli_runtime_error.h"
#include <iostream>
#include <sstream>
#if _DEBUG
#pragma comment(lib, "FlyCapture2d_v140.lib")
#else
#pragma comment(lib, "FlyCapture2_v140.lib")
#endif

void print_error(const FlyCapture2::Error& error)
{
	error.PrintErrorTrace();
}

#define FLYCAM_SAFE_CALL(err) fly_cam_safe_call(err,__FILE__,__LINE__)
inline void fly_cam_safe_call(const FlyCapture2::Error& error, const char* file, const int line)
{
	if (error != FlyCapture2::PGRERROR_OK)
	{
		std::stringstream ss;
		ss << file << ":" << line << std::endl;
		print_error(error);
		qli_runtime_error(ss.str());
	}
}

auto default_format = FlyCapture2::PIXEL_FORMAT_RAW16;

FlyCapture2::Format7ImageSettings aoi_to_fmt7_image_settings(const camera_aoi& aoi, const FlyCapture2::PixelFormat& k_fmt7_pix_fmt)
{
	const static auto k_fmt7_mode = FlyCapture2::MODE_0;
	FlyCapture2::Format7ImageSettings fmt_image_settings;
	fmt_image_settings.mode = k_fmt7_mode;
	fmt_image_settings.offsetX = aoi.left;
	fmt_image_settings.offsetY = aoi.top;
	fmt_image_settings.width = aoi.width;
	fmt_image_settings.height = aoi.height;
	fmt_image_settings.pixelFormat = k_fmt7_pix_fmt;
	return fmt_image_settings;
}

bool validate_fmt7_image_settings(FlyCapture2::Camera* cam, const FlyCapture2::Format7ImageSettings& flycap_settings)
{
	FlyCapture2::Format7PacketInfo fmt7_packet_info;
	bool valid;
	cam->ValidateFormat7Settings(&flycap_settings, &valid, &fmt7_packet_info);
	return valid;
}

FlyCapture2::Property get_property(FlyCapture2::Camera* cam, const FlyCapture2::PropertyType type)
{//can't forward declare an enum
	FlyCapture2::Property prop;
	prop.type = type;
	cam->GetProperty(&prop);
	return prop;
}

void fly_capture_device::wait_for_power_on() const
{
	const unsigned int k_camera_power = 0x610;
	const auto k_power_val = 0x80000000;
	FLYCAM_SAFE_CALL(cam_->WriteRegister(k_camera_power, k_power_val));
	//
	const auto milliseconds_to_sleep = ms_to_chrono(100);
	unsigned int reg_val = 0;
	unsigned int retries = 10;
	do
	{
		windows_sleep(milliseconds_to_sleep);
		const auto error = cam_->ReadRegister(k_camera_power, &reg_val);
		if (error == FlyCapture2::PGRERROR_TIMEOUT)
		{
			// ignore timeout errors, camera may not be responding to
			// register reads during power-up
		}
		else if (error != FlyCapture2::PGRERROR_OK)
		{
			print_error(error);
			qli_runtime_error();
		}

		retries--;
	} while ((reg_val & k_power_val) == 0 && retries > 0);
}

void fly_capture_device::poll_for_trigger_ready() const
{
	const unsigned int k_software_trigger = 0x62C;
	unsigned int reg_val = 0;
	do
	{
		const auto error = cam_->ReadRegister(k_software_trigger, &reg_val);
		if (error != FlyCapture2::PGRERROR_OK)
		{
			print_error(error);
			qli_runtime_error("Error polling for trigger, aka trigger didn't activate");
		}

	} while (reg_val >> 31 != 0);
}

void fly_capture_device::fire_software_trigger() const
{
	const unsigned int k_software_trigger = 0x62C;
	const auto k_fire_val = 0x80000000;
	FLYCAM_SAFE_CALL(cam_->WriteRegister(k_software_trigger, k_fire_val));
}

fly_capture_device::fly_capture_device(const int camera_idx, QObject* parent) : camera_device(camera_device_features(false, false, false, false, camera_contrast_features(camera_chroma::monochrome, demosaic_mode::no_processing, { 70,65535 })), camera_idx, parent), cam_(nullptr)
{
	cam_ = new FlyCapture2::Camera();
	FlyCapture2::BusManager bus_mgr;
	unsigned int num_cameras;
	FLYCAM_SAFE_CALL(bus_mgr.GetNumOfCameras(&num_cameras));
	FlyCapture2::PGRGuid guid;
	FLYCAM_SAFE_CALL(bus_mgr.GetCameraFromIndex(0, &guid));
	FLYCAM_SAFE_CALL(cam_->Connect(&guid));
	wait_for_power_on();
	FlyCapture2::CameraInfo cam_info;
	FLYCAM_SAFE_CALL(cam_->GetCameraInfo(&cam_info));
	chroma = cam_info.isColorCamera ? camera_chroma::optional_color : camera_chroma::monochrome;
	//demosaic_mode_modes = { cam_info.isColorCamera ? demosaic_mode::rggb_14_native : demosaic_mode::no_processing };
	FlyCapture2::TriggerModeInfo trigger_mode_info;
	FLYCAM_SAFE_CALL(cam_->GetTriggerModeInfo(&trigger_mode_info));
	//
	if (!trigger_mode_info.present)
	{
		qli_runtime_error("Camera does not support external trigger! Exiting...");
	}
	//Camera will have the last powered on state, thus we set it to the state for our program
	const auto make_disabled_prop = [&](const FlyCapture2::PropertyType type, const bool has_absolute_control) {
		auto prop = get_property(cam_, type);
		prop.type = type;
		prop.absControl = has_absolute_control ? true : prop.absControl;
		prop.onOff = false;
		prop.autoManualMode = false;
		return prop;
	};
	{
		const auto prop = make_disabled_prop(FlyCapture2::BRIGHTNESS, true);
		FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
	}
	{
		const auto prop = make_disabled_prop(FlyCapture2::AUTO_EXPOSURE, true);
		FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
	}
	{
		const auto prop = make_disabled_prop(FlyCapture2::SHARPNESS, false);
		FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
	}
	if (cam_info.isColorCamera)
	{
		{
			auto prop = make_disabled_prop(FlyCapture2::WHITE_BALANCE, false);
			prop.valueA = 1;
			prop.valueB = 1;
			FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
		}
		{
			const auto prop = make_disabled_prop(FlyCapture2::HUE, true);
			FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
		}
		{
			const auto prop = make_disabled_prop(FlyCapture2::SATURATION, true);
			FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
		}
	}
	{
		auto prop = make_disabled_prop(FlyCapture2::GAMMA, true);
		prop.absValue = 10.0;
		FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
	}
	//Parallel the readout, might lead to synchronization problems;;;
	{
		FlyCapture2::TriggerMode trigger_mode;
		FLYCAM_SAFE_CALL(cam_->GetTriggerMode(&trigger_mode));
		// Set camera to trigger mode 0
		trigger_mode.onOff = true;
		trigger_mode.mode = 14;
		trigger_mode.parameter = 0;
		trigger_mode.source = 7;// A source of 7 means software trigger
		FLYCAM_SAFE_CALL(cam_->SetTriggerMode(&trigger_mode));
		poll_for_trigger_ready();
	}
	{
		FlyCapture2::FC2Config config;
		FLYCAM_SAFE_CALL(cam_->GetConfiguration(&config));
		config.grabTimeout = 5000;
		config.numBuffers = get_internal_buffer_count();
		config.grabMode = FlyCapture2::BUFFER_FRAMES;
#if _DEBUG
		const auto do_high_performance = true;
#else
		const auto do_high_performance = false;
#endif
		config.highPerformanceRetrieveBuffer = do_high_performance;
		FLYCAM_SAFE_CALL(cam_->SetConfiguration(&config));
	}
	//build AOIs etc
	{
		bin_modes.emplace_back(camera_bin(1));
		//
		FlyCapture2::Format7Info fmt7_info;
		bool p_supported;
		cam_->GetFormat7Info(&fmt7_info, &p_supported);
		aois = {
		camera_aoi(fmt7_info.maxWidth, fmt7_info.maxHeight, 0, 0),
		camera_aoi(1920, 1080, 128, 4),
		camera_aoi(1440, 1080, 184, 244),
		camera_aoi(1928, 512, 468, 0),
		camera_aoi(1024, 1024, 212, 452),
		camera_aoi(768, 768, 340, 580),
		camera_aoi(512, 512, 468, 708),
		camera_aoi(256, 256, 596, 836)
		};
		for (auto&& aoi : aois)
		{
			aoi.re_center_and_fixup(fmt7_info.maxWidth, fmt7_info.maxHeight);
		}
		aois.erase(
			std::remove_if(aois.begin(), aois.end(),
				[&](const camera_aoi& o)
		{
			const auto as_format_seven = aoi_to_fmt7_image_settings(o, default_format);
			const auto good = validate_fmt7_image_settings(cam_, as_format_seven);
			return !good;
		}),
			aois.end());
		//
	}
	common_post_constructor();
}

void fly_capture_device::trigger_internal()
{
	poll_for_trigger_ready();
	fire_software_trigger();
}

bool  fly_capture_device::capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds&)
{
	//timeout ignored
	FlyCapture2::Image image;
	const auto error_msg = cam_->RetrieveBuffer(&image);
	if (error_msg != FlyCapture2::PGRERROR_OK)
	{
		std::cout << "FlyCapture failed to get buffer" << std::endl;
		return false;
	}
	const auto stamp = image.GetTimeStamp();
	const auto timestamp_s = stamp.seconds * 1.0 + stamp.microSeconds / (1000.0 * 1000.0);
	unsigned int p_rows, p_cols, p_stride;
	FlyCapture2::PixelFormat p_pixel_format;
	FlyCapture2::BayerTileFormat p_bayer_format;//used for color interlacing
	image.GetDimensions(&p_rows, &p_cols, &p_stride, &p_pixel_format, &p_bayer_format);
	const auto raw_ptr = reinterpret_cast<unsigned short*>(image.GetData());
	//auto bytes = image.GetDataSize();//some kind of error checking here
	const auto received_frame_info = frame_size(p_cols, p_rows);
	//
	const auto timestamp = std::chrono::duration<double, std::ratio<1> >(timestamp_s);
	const auto timestamp_ms = std::chrono::duration_cast<std::chrono::microseconds>(timestamp);
	const frame_meta_data meta_data_after(meta_data, timestamp_ms);
	const auto info = image_info(received_frame_info,1,image_info::complex::no);
	const camera_frame<unsigned short> received_frame(raw_ptr, info, meta_data_after);
	fill_me(received_frame);
	return true;
}

void fly_capture_device::set_cooling_internal(bool)
{
	qli_not_implemented();
}

void fly_capture_device::fix_camera_internal()
{
	windows_sleep(ms_to_chrono(500));
}

void fly_capture_device::apply_settings_internal(const camera_config& new_config)
{
	//does nothing,for now?
	auto& aoi = aois.at(new_config.aoi_index);
	auto flycap_settings = aoi_to_fmt7_image_settings(aoi, default_format);
	const auto valid = validate_fmt7_image_settings(cam_, flycap_settings);
	if (!valid)
	{
		qli_runtime_error();
	}
	const FlyCapture2::Format7PacketInfo fmt7_packet_info;
	FLYCAM_SAFE_CALL(cam_->SetFormat7Configuration(&flycap_settings, fmt7_packet_info.recommendedBytesPerPacket));
}

void fly_capture_device::set_exposure_internal(const std::chrono::microseconds& exposure)
{
	const auto exposure_ms = std::chrono::duration_cast<std::chrono::milliseconds>(exposure);
	auto prop = get_property(cam_, FlyCapture2::SHUTTER);
	prop.absControl = true;
	prop.autoManualMode = false;
	prop.onOff = false;
	prop.absValue = exposure_ms.count();//in ms default
	FLYCAM_SAFE_CALL(cam_->SetProperty(&prop));
}

void fly_capture_device::print_debug(std::ostream&)
{
	//
}

void fly_capture_device::start_capture_internal()
{
	FLYCAM_SAFE_CALL(cam_->StartCapture());
}

void fly_capture_device::stop_capture_internal()
{
	FLYCAM_SAFE_CALL(cam_->StopCapture());
}


std::chrono::microseconds fly_capture_device::get_min_exp_internal()
{
	return ms_to_chrono(1);
}


fly_capture_device::~fly_capture_device()
{
	FLYCAM_SAFE_CALL(cam_->Disconnect());
	delete cam_;
}

std::chrono::microseconds fly_capture_device::get_min_cycle_internal()
{
	//FPS
	const auto prop = get_property(cam_, FlyCapture2::FRAME_RATE);
	const auto frame_rate = prop.absValue;
	return ms_to_chrono((1000 / frame_rate) + 1);
}

#endif