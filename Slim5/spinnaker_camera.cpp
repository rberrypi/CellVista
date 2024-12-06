#include "stdafx.h"
#if CAMERA_PRESENT_SPINRAKER == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "spinnaker_camera.h"
#if _DEBUG
#pragma comment(lib,"Spinnakerd_v140.lib")
#else
#pragma comment(lib,"Spinnaker_v140.lib")
#endif
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include "qli_runtime_error.h"

struct spinnaker_camera_impl
{
	Spinnaker::SystemPtr system_ptr;
	Spinnaker::CameraPtr camera_ptr;
};

inline void spinnaker_cam_safe_call_error(const char* file, const int line)
{
	std::stringstream ss;
	ss << "Spinnaker Error: " << file << ":" << line;
	qli_runtime_error(ss.str());
}

inline void spinnaker_cam_safe_call_error(const char* msg, const char* file, const int line)
{
	std::cout << "SDK Said: " << msg << std::endl << std::flush;
	spinnaker_cam_safe_call_error(file, line);
}


#define SPINNAKER_ERROR_MESSAGE( err ) spinnaker_cam_safe_call_error(err, __FILE__, __LINE__ )

void spinnaker_camera::debug_fps(const char* file, const int line) const
{
	Spinnaker::GenApi::INodeMap& node_map = impl->camera_ptr->GetNodeMap();
	Spinnaker::GenApi::CFloatPtr ptrAcquisitionResultingFrameRateHz = node_map.GetNode("AcquisitionResultingFrameRate");
	const auto frame_rate_hz = ptrAcquisitionResultingFrameRateHz->GetValue();
	std::cout << file << "::" << line << " -> FPS " << frame_rate_hz << std::endl;
}
#define DEBUG_FPS() debug_fps(__FILE__,__LINE__)

spinnaker_camera::~spinnaker_camera()
{
	//out of the kindness of my heart I'll also reset the trigger?
	{
		auto& nodeMap = impl->camera_ptr->GetNodeMap();
		Spinnaker::GenApi::CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
		if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
		{
			SPINNAKER_ERROR_MESSAGE("Unable to disable trigger mode (node retrieval). Non-fatal error...");
		}

		Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
		if (!IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
		{
			SPINNAKER_ERROR_MESSAGE("Unable to disable trigger mode (enum entry retrieval). Non-fatal error...");
		}
		ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());
	}
	impl->camera_ptr->DeInit();
	impl->camera_ptr = 0;//important because it frees the reference counting, the reference counting here is kinda bullshit
	impl->system_ptr->ReleaseInstance();
}

void spinnaker_camera::switch_to_software_trigger() const
{
	Spinnaker::GenApi::INodeMap& node_map = impl->camera_ptr->GetNodeMap();
	//

	Spinnaker::GenApi::CEnumerationPtr ptrTriggerMode = node_map.GetNode("TriggerMode");
	if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to disable trigger mode (node retrieval). Aborting...");
	}
	Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
	if (!Spinnaker::GenApi::IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to disable trigger mode (enum entry retrieval). Aborting...");
	}
	ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());
	DEBUG_FPS();
	//
	Spinnaker::GenApi::CEnumerationPtr ptrTriggerSource = node_map.GetNode("TriggerSource");
	if (!IsAvailable(ptrTriggerSource) || !IsWritable(ptrTriggerSource))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to set trigger mode (node retrieval). Aborting...");
	}
	//
	Spinnaker::GenApi::CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");
	if (!IsAvailable(ptrTriggerSourceSoftware) || !IsReadable(ptrTriggerSourceSoftware))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to set trigger mode (enum entry retrieval). Aborting...");
	}
	ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());
	DEBUG_FPS();
	//
	Spinnaker::GenApi::CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
	if (!IsAvailable(ptrTriggerModeOn) || !IsReadable(ptrTriggerModeOn))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to enable trigger mode (enum entry retrieval). Aborting...");
	}
	ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());
	DEBUG_FPS();
	//Required 1 second sleep for Blackfly cameras
	windows_sleep(std::chrono::seconds(2));
}

int spinnaker_camera::get_internal_buffer_count() const
{
	return 30;//some assumption, don't have camera to test yet
}


void spinnaker_camera::flush_camera_internal_buffer()
{
	qli_not_implemented();
}

void spinnaker_camera::fix_camera_internal()
{
	qli_not_implemented();
}

void spinnaker_camera::apply_settings_internal(const camera_config& new_config)
{
	auto& nodeMap = impl->camera_ptr->GetNodeMap();
	{
		Spinnaker::GenApi::CEnumerationPtr ptrGainAuto = nodeMap.GetNode("GainAuto");
		try
		{
			ptrGainAuto->SetIntValue(Spinnaker::GainAuto_Off);
		}
		catch (Spinnaker::Exception& error)
		{
			SPINNAKER_ERROR_MESSAGE(error.what());
		}
		DEBUG_FPS();
	}
	{
		try
		{
			impl->camera_ptr->GammaEnable.SetValue(false);
		}
		catch (Spinnaker::Exception& error)
		{
			SPINNAKER_ERROR_MESSAGE(error.what());
		}
		DEBUG_FPS();
	}
	//perhaps investigate black level clamping
	{
		//Spinnaker::GenApi::CEnumerationPtr ptrBlackLevelClamping = nodeMap.GetNode("BlackLevelClampingEnable");
	}
	{
		Spinnaker::GenApi::CEnumerationPtr AdcBitDepth = nodeMap.GetNode("AdcBitDepth");
		//set the highest bit depth
		//programatically get these items was too hard
		for (auto try_this : { Spinnaker::AdcBitDepth_Bit14,Spinnaker::AdcBitDepth_Bit12,Spinnaker::AdcBitDepth_Bit10,Spinnaker::AdcBitDepth_Bit8 })
		{
			try
			{
				AdcBitDepth->SetIntValue(try_this);
				break;
			}
			catch (Spinnaker::Exception&)
			{
			}
		}
		DEBUG_FPS();
	}
	switch_to_software_trigger();
	//sets the pixel format
	{
		Spinnaker::GenApi::CEnumerationPtr ptrPixelFormat = nodeMap.GetNode("PixelFormat");
		const auto current_format = impl->camera_ptr->PixelFormat.GetCurrentEntry();
		std::cout << "Current Pixel Format: " << current_format->GetSymbolic() << std::endl;
		if (!(IsAvailable(ptrPixelFormat)))
		{
			SPINNAKER_ERROR_MESSAGE("Pixel Format Not Available");
		}
		Spinnaker::GenApi::CEnumEntryPtr ptrPixelFormatMono16 = ptrPixelFormat->GetEntryByName("Mono16");
		const auto mono_sixteen_value = ptrPixelFormatMono16->GetValue();
		const auto change_format = current_format->GetValue() != mono_sixteen_value;
		if (change_format)
		{
			if (!IsWritable(ptrPixelFormat))
			{
				SPINNAKER_ERROR_MESSAGE("Can't set the pixel format to Mono16");
			}
			ptrPixelFormat->SetIntValue(mono_sixteen_value);
			DEBUG_FPS();
		}
	}
	//Set size
	const auto& size_to_set = aois[new_config.aoi_index];
	{
		// must be done in a certain order!
		const auto set_node = [&](auto name, auto value)
		{
			Spinnaker::GenApi::CIntegerPtr node = nodeMap.GetNode(name);
			if (!(IsAvailable(node) && IsWritable(node)))
			{
				SPINNAKER_ERROR_MESSAGE("Can't set sensor size");
			}
			node->SetValue(value);
		};
		set_node("Width", size_to_set.width);
		set_node("Height", size_to_set.height);
		set_node("OffsetX", size_to_set.left);
		set_node("OffsetY", size_to_set.top);
		DEBUG_FPS();
	}
	//Kill Auto Contrast
	{
		Spinnaker::GenApi::CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
		if (!IsAvailable(ptrExposureAuto) || !IsWritable(ptrExposureAuto))
		{
			SPINNAKER_ERROR_MESSAGE("Unable to set Exposure Auto (enumeration retrieval). Aborting...");
		}
		Spinnaker::GenApi::CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
		if (!IsAvailable(ptrExposureAutoOff) || !IsReadable(ptrExposureAutoOff))
		{
			SPINNAKER_ERROR_MESSAGE("Unable to set Exposure Auto (entry retrieval). Aborting...");
		}
		const auto exposure_auto_off = ptrExposureAutoOff->GetValue();
		ptrExposureAuto->SetIntValue(exposure_auto_off);
		DEBUG_FPS();
	}
	// Set to continuous
	{
		Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
		if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
		{
			SPINNAKER_ERROR_MESSAGE("Unable to set acquisition mode to continuous(node retrieval).Aborting...");
		}
		Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
		if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
		{
			SPINNAKER_ERROR_MESSAGE("Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting...");
		}
		const auto acquisition_mode_continuous = ptrAcquisitionModeContinuous->GetValue();
		ptrAcquisitionMode->SetIntValue(acquisition_mode_continuous);
		DEBUG_FPS();
	}
}

camera_contrast_features::demosaic_modes_set spinnaker_demosaic_modes = { demosaic_mode::polarization_0_45_90_135, demosaic_mode::polarization_0_90, demosaic_mode::polarization_45_135 };
spinnaker_camera::spinnaker_camera(const int camera_idx, QObject* parent) : camera_device(camera_device_features(false, false, false, false, camera_contrast_features(camera_chroma::monochrome, spinnaker_demosaic_modes, { 0,65535 })), camera_idx, parent)
{
	impl = std::make_unique<spinnaker_camera_impl>();
	impl->system_ptr = Spinnaker::System::GetInstance();
	const auto camera_list = impl->system_ptr->GetCameras();
	const auto cameras_detected = camera_list.GetSize();
	if (cameras_detected == 0)
	{
		SPINNAKER_ERROR_MESSAGE("No Spinnaker SDK Cameras Found");
	}
	const auto default_camera_selection = 0;
	impl->camera_ptr = camera_list.GetByIndex(default_camera_selection);
	if (!impl->camera_ptr->IsValid())
	{
		SPINNAKER_ERROR_MESSAGE("Camera not valid for use, for some reason");
	}
	impl->camera_ptr->Init();
	if (!impl->camera_ptr->IsInitialized())
	{
		SPINNAKER_ERROR_MESSAGE("Could not initialize camera");
	}
	windows_sleep(std::chrono::seconds(3));
	DEBUG_FPS();
	{
		auto& nodeMap = impl->camera_ptr->GetNodeMap();
		const auto get_limits = [&](const auto value)
		{
			Spinnaker::GenApi::CIntegerPtr node = nodeMap.GetNode(value);
			if (!(IsAvailable(node)))
			{
				SPINNAKER_ERROR_MESSAGE("Can't get sensor size");
			}
			return node->GetMax();
		};
		const auto width_range = get_limits("Width");
		const auto height_range = get_limits("Height");
		//todo generate more of these
		const auto steps = 6;
		for (auto sy = 1; sy < steps; ++sy)
		{
			for (auto sx = 1; sx < steps; ++sx)
			{
				auto aoi = camera_aoi(width_range * static_cast<float>(sx) / steps, height_range * static_cast<float>(sy) / steps, 0, 0);
				aoi.re_center_and_fixup(width_range, height_range, 4);
				aois.push_back(aoi);
			}
		}
		bin_modes.emplace_back(camera_bin(1));
		//
		//detection for the chroma and polariation goes here, right now its hard coded.
	}
	//
	//
	common_post_constructor();
}

void spinnaker_camera::start_capture_internal()
{
	try
	{
		impl->camera_ptr->BeginAcquisition();
	}
	catch (Spinnaker::Exception& ptr)
	{
		std::cout << ptr.what() << std::endl;
		throw;
	}
}

void spinnaker_camera::stop_capture_internal()
{
	try
	{
		impl->camera_ptr->EndAcquisition();
	}
	catch (Spinnaker::Exception& ptr)
	{
		std::cout << ptr.what() << std::endl;
		throw;
	}
}

void spinnaker_camera::trigger_internal()
{
	auto& nodeMap = impl->camera_ptr->GetNodeMap();
	Spinnaker::GenApi::CCommandPtr ptrSoftwareTriggerCommand = nodeMap.GetNode("TriggerSoftware");
	if (!IsAvailable(ptrSoftwareTriggerCommand) || !IsWritable(ptrSoftwareTriggerCommand))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to execute trigger. Aborting...");
	}
	ptrSoftwareTriggerCommand->Execute();
	// "Blackfly and Flea3 GEV cameras need 2 second delay after software trigger (?)" Is this some kind of nonsense?
}

bool spinnaker_camera::capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds&)
{
	//timeout is ignored, thing will lock if something goes wrong
	auto pResultImage = impl->camera_ptr->GetNextImage();
	if (pResultImage->IsIncomplete())
	{
		SPINNAKER_ERROR_MESSAGE("Received an incomplete image, this shouldn't happen");
	}
	const auto buffer_size = pResultImage->GetBufferSize();
	const auto& roi = aois[camera_configuration_.aoi_index];
	const auto buffer_size_expected = roi.n() * sizeof(unsigned short);
	//const auto per_pixel = pResultImage->GetBitsPerPixel();
	//const auto width = pResultImage->GetWidth();
	//const auto height = pResultImage->GetHeight();
	//const auto get_stride = pResultImage->GetStride();
	const auto timestamp_nano_seconds = pResultImage->GetTimeStamp();//What?
	const auto timestamp_nano_seconds_chrono = std::chrono::duration<uint64_t, std::nano>(timestamp_nano_seconds);
	const auto timestamp_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(timestamp_nano_seconds_chrono);
	if (buffer_size < buffer_size_expected)
	{
		SPINNAKER_ERROR_MESSAGE("Wrong number of bits received, go download more bits");
	}
	auto* ptr = static_cast<unsigned short*>(pResultImage->GetData());
	frame_meta_data meta_data_after_acquire(meta_data,timestamp_microseconds);
	const auto info = image_info(roi.to_frame_size(),1,image_info::complex::no);
	camera_frame<unsigned short> frame(ptr, info, meta_data_after_acquire);
	fill_me(frame);
	pResultImage->Release();
	return true;//no chance for error :-)
}

void spinnaker_camera::set_cooling_internal(bool) 
{
	qli_not_implemented();
}

void spinnaker_camera::print_debug(std::ostream&)
{
	qli_not_implemented();
}

void spinnaker_camera::set_exposure_internal(const std::chrono::microseconds& exposure)
{
	Spinnaker::GenApi::INodeMap& node_map = impl->camera_ptr->GetNodeMap();
	Spinnaker::GenApi::CFloatPtr ptrExposureTime = node_map.GetNode("ExposureTime");
	if (!IsAvailable(ptrExposureTime) || !IsWritable(ptrExposureTime))
	{
		SPINNAKER_ERROR_MESSAGE("Unable to set Exposure Time (float retrieval). Aborting...");
	}
	const auto exposure_microseconds = exposure.count();
	try
	{
		ptrExposureTime->SetValue(exposure_microseconds);
	}
	catch (const Spinnaker::Exception& ptr)
	{
		std::cout << ptr.what() << std::endl;
		throw;
	}
}

[[nodiscard]] QStringList spinnaker_camera::get_gain_names_internal() const 
{
	QStringList none;
	none <<"none";
	return none;
}

std::chrono::microseconds spinnaker_camera::get_min_exp_internal()
{
	Spinnaker::GenApi::INodeMap& node_map = impl->camera_ptr->GetNodeMap();
	Spinnaker::GenApi::CFloatPtr ptrExposureTime = node_map.GetNode("ExposureTime");
	if (IsAvailable(ptrExposureTime))
	{
		const size_t min_exp_in_us = ptrExposureTime->GetMin();
		return std::chrono::microseconds(min_exp_in_us);
	}
	else
	{
		return ms_to_chrono(1);
	}
}

std::chrono::microseconds spinnaker_camera::get_min_cycle_internal()
{
	Spinnaker::GenApi::INodeMap& node_map = impl->camera_ptr->GetNodeMap();
	Spinnaker::GenApi::CFloatPtr ptrAcquisitionResultingFrameRateHz = node_map.GetNode("AcquisitionResultingFrameRate");
	const auto frame_rate_hz = ptrAcquisitionResultingFrameRateHz->GetValue();
	const auto fudge = ms_to_chrono((1000 / frame_rate_hz));
	return fudge;
}


#endif