#include "stdafx.h"
#include <stdafx.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#if CAMERA_PRESENT_PCO_PANDA == CAMERA_PRESENT || \
	CAMERA_PRESENT_PCO_EDGE == CAMERA_PRESENT || \
	BUILD_ALL_DEVICES_TARGETS

#include <pco.h>

// Static link libraries
#pragma comment(lib, "SC2_Cam.lib")


void __pco_err_chk(const int err_code, const char* file, const int line) {
	if (err_code != PCO_NOERROR)
	{
		std::stringstream runtime_error_stringstream;

		const bool is_error = 0x80000000 & err_code; // for error, only the error bit is set
		const bool is_warning = is_error && (err_code & 0x40000000); // for warning, both the error and the warning bit will be set.

		char err_string[1024];
		PCO_GetErrorTextSDK(err_code, err_string, 1024);

		if (is_warning)
		{
			runtime_error_stringstream << "WARNING: " << file << ":" << line;
			runtime_error_stringstream << ": " << err_string << '\n';
			std::cout << runtime_error_stringstream.str() << std::endl;
		}

		else if (is_error)
		{
			runtime_error_stringstream << "ERROR: " << file << ":" << line;
			runtime_error_stringstream << ": " << err_string << '\n';
			qli_runtime_error(runtime_error_stringstream.str());
		}
	}
}


void pco::camera_internal_buffer::allocate(HANDLE& cam_handle, size_t xResAct, size_t yResAct) {
	for (uint64_t i = 0; i < _num_buffers; ++i) {
		LOGGER_INFO("allocating internal: " << i);
		PCO_ERR_CHK(PCO_AllocateBuffer(cam_handle,
			BufNum + i,
			xResAct * yResAct * sizeof(WORD),
			BufAdr + i,
			BufEvent + i));
	}
}


void pco::camera_internal_buffer::free_buffers_in_camera(const HANDLE& cam_handle) const {
	for (size_t i = 0; i < _num_buffers; ++i)
		if (BufAdr[i] != (WORD*)0) {
			LOGGER_INFO("freeing: " << i << " -> " << BufAdr[i]);
			PCO_ERR_CHK(PCO_FreeBuffer(cam_handle, (SHORT)i));
			BufNum[i] = (SHORT)(-1); // we might want to create new buffers later on
			BufAdr[i] = (WORD*)0;
		}
}


pco::pco(const camera_device_features& cdf, size_t buf_size, int camera_id, QObject* parent) :
	camera_device(cdf, camera_id, parent),
	_buffer(buf_size)
{
	// zero out all the structs. This is to prevent
	// garbage values from interfering 
	memset(&strGeneral, 0, sizeof(PCO_General));
	memset(&strCamType, 0, sizeof(PCO_CameraType));
	memset(&strSensor, 0, sizeof(PCO_Sensor));
	memset(&strDescription, 0, sizeof(PCO_Description));
	memset(&strTiming, 0, sizeof(PCO_Timing));
	memset(&strStorage, 0, sizeof(PCO_Storage));
	memset(&strRecording, 0, sizeof(PCO_Recording));

	// fill the size of all PCO Structures
	strGeneral.wSize = sizeof(strGeneral);
	strGeneral.strCamType.wSize = sizeof(strGeneral.strCamType);
	strCamType.wSize = sizeof(strCamType);
	strSensor.wSize = sizeof(strSensor);
	strSensor.strDescription.wSize = sizeof(strSensor.strDescription);
	strSensor.strDescription2.wSize = sizeof(strSensor.strDescription2);
	strDescription.wSize = sizeof(strDescription);
	strTiming.wSize = sizeof(strTiming);
	strStorage.wSize = sizeof(strStorage);
	strRecording.wSize = sizeof(strRecording);

	// open a connection to the camera
	PCO_ERR_CHK(PCO_OpenCamera(&cam_handle, 0));

	// if previous recording state was set, unset it now
	PCO_ERR_CHK(PCO_GetRecordingState(cam_handle, &RecordingState));
	if (RecordingState) PCO_ERR_CHK(PCO_SetRecordingState(cam_handle, 0));

	//set camera to default state
	PCO_ERR_CHK(PCO_ResetSettingsToDefault(cam_handle));

	PCO_ERR_CHK(PCO_ArmCamera(cam_handle));
}

pco::~pco() {
	_buffer.free_buffers_in_camera(cam_handle);
	PCO_ERR_CHK(PCO_CancelImages(cam_handle));
	PCO_ERR_CHK(PCO_CloseCamera(cam_handle));
}


void pco::add_bin_modes() {
	if (strDescription.wBinHorzSteppingDESC == 0)  // steps multiple of 2		
		for (WORD i = 1; i <= strDescription.wMaxBinHorzDESC; i *= 2)
			bin_modes.push_back(camera_bin(i));

	else if (strDescription.wBinHorzSteppingDESC == 1)  // steps increment by 1
		for (WORD i = 1; i <= strDescription.wMaxBinHorzDESC; ++i)
			bin_modes.push_back(camera_bin(i));
}


void pco::init() {
	// fill the descriptor structures 
	fill_pco_descriptors();

	// Initialize the area of interests (aois) and bin modes.
	add_bin_modes();
	initialize_aois();

	common_post_constructor();
}


void pco::fill_pco_descriptors() {
	PCO_ERR_CHK(PCO_GetGeneral(cam_handle, &strGeneral));
	PCO_ERR_CHK(PCO_GetCameraType(cam_handle, &strCamType));
	PCO_ERR_CHK(PCO_GetSensorStruct(cam_handle, &strSensor));
	PCO_ERR_CHK(PCO_GetCameraDescription(cam_handle, &strDescription));
	PCO_ERR_CHK(PCO_GetTimingStruct(cam_handle, &strTiming));
	PCO_ERR_CHK(PCO_GetStorageStruct(cam_handle, &strStorage));
	PCO_ERR_CHK(PCO_GetRecordingStruct(cam_handle, &strRecording));
}


void pco::flush_camera_internal_buffer() {
	PCO_ERR_CHK(PCO_ArmCamera(cam_handle));
}


QStringList pco::get_gain_names_internal() const
{
	return QStringList("None");
}


void pco::set_exposure_internal(const std::chrono::microseconds& exposure)
{
	cout << "set_exposure_internal" << endl;
	DWORD dwDelay, dwExposure, _exposure_ns;

	WORD wTimeBaseDelay, wTimeBaseExposure;
	PCO_ERR_CHK(PCO_GetDelayExposureTime(cam_handle, &dwDelay, &dwExposure, &wTimeBaseDelay, &wTimeBaseExposure));

	if (wTimeBaseExposure == 0x0000) // nanosecond
		_exposure_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(exposure).count();

	else if (wTimeBaseExposure == 0x0001) // microsecond
		_exposure_ns = exposure.count();

	else if (wTimeBaseExposure == 0x0002) // millisecond
		_exposure_ns = std::chrono::duration_cast<std::chrono::milliseconds>(exposure).count();

	PCO_ERR_CHK(PCO_SetDelayExposureTime(cam_handle, dwDelay, _exposure_ns, wTimeBaseDelay, wTimeBaseExposure));
}

void pco::print_debug(std::ostream& input)
{
	qli_not_implemented();
}

std::chrono::microseconds pco::get_transfer_time_internal()
{
	return ns_to_us_chrono(strDescription.dwMinDelayDESC);
}

std::chrono::microseconds pco::get_min_exposure_internal()
{
//	return ns_to_us_chrono(std::max(strDescription.dwMinExposureDESC, strDescription.dwMinExposureIRDESC));
	return ns_to_us_chrono(1);
}


std::chrono::microseconds pco::get_readout_time_internal()
{
	DWORD dwTime_s, dwTime_ns;
	PCO_ERR_CHK(PCO_GetCOCRuntime(cam_handle, &dwTime_s, &dwTime_ns));
	return ns_to_us_chrono(dwTime_ns);
}

void pco::start_capture_internal()
{
	// set the image parameters for internally allocated resources
	PCO_ERR_CHK(PCO_SetImageParameters(cam_handle,
		XResAct,
		YResAct,
		IMAGEPARAMETERS_READ_WHILE_RECORDING,
		NULL,
		0));

	// ask the camera to start recording
	PCO_ERR_CHK(PCO_GetRecordingState(cam_handle, &RecordingState));
	if (!RecordingState) PCO_ERR_CHK(PCO_SetRecordingState(cam_handle, 1));

	PCO_ERR_CHK(PCO_SetTimestampMode(cam_handle, 0x0001));
}

void pco::stop_capture_internal()
{
	PCO_ERR_CHK(PCO_GetRecordingState(cam_handle, &RecordingState));
	if (RecordingState) PCO_ERR_CHK(PCO_SetRecordingState(cam_handle, 0));
}

bool pco::capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator,
	std::vector<capture_item>::const_iterator>& frames,
	const frame_meta_data_before_acquire& meta_data,
	const std::chrono::microseconds& exposure,
	const std::chrono::microseconds& frame_time_out,
	const camera_frame_processing_function& process_function)
{
	qli_not_implemented();
	return false;
}

void pco::set_cooling_internal(bool enable)
{
	qli_not_implemented();
}

/**
 * @brief This function will only get triggered if the capture internal method failed.
*/
void pco::fix_camera_internal()
{
	_buffer.free_buffers_in_camera(cam_handle);

	//set camera to default state
	PCO_ERR_CHK(PCO_ResetSettingsToDefault(cam_handle));

	// send the modified settings to the camera
	PCO_ERR_CHK(PCO_ArmCamera(cam_handle));

	fill_pco_descriptors();

	DWORD CameraWarning, CameraError, CameraStatus;
	PCO_ERR_CHK(PCO_GetCameraHealthStatus(cam_handle, &CameraWarning, &CameraError, &CameraStatus));
	if (CameraError != 0)
	{
		PCO_ERR_CHK(PCO_CloseCamera(cam_handle));
		PCO_ERR_CHK(CameraError);
	}

	// sets the image parameters for internally allocated resources.
	PCO_ERR_CHK(PCO_SetImageParameters(cam_handle, XResAct, YResAct, IMAGEPARAMETERS_READ_WHILE_RECORDING, 0, 0));
}

void pco::trigger_internal() {}

bool pco::capture_internal(const camera_frame_processing_function& fill_me,
	const frame_meta_data_before_acquire& meta_data,
	const std::chrono::microseconds& timeout)
{
	PCO_ERR_CHK(PCO_GetImageEx(cam_handle, strStorage.wActSeg, 0, 0, 0, XResAct, YResAct, 16));
	this->publish_frame(fill_me, meta_data, 0);
	return true; // if an error occurs, it will throw an error and terminate the program
}

void pco::apply_settings_internal(const camera_config& new_config)
{
	// Free previously allocated buffer
	_buffer.free_buffers_in_camera(cam_handle);

	// if previous recording state was set, unset it now or else the
	// new settings will be rejected
	PCO_ERR_CHK(PCO_GetRecordingState(cam_handle, &RecordingState));
	if (RecordingState) PCO_ERR_CHK(PCO_SetRecordingState(cam_handle, 0));

	// make sure that the binned images take up the entire drawable space
	// this is kind of a hack around the codebase. This is also to prevent
	// CUDA from throwing a segmentation fault, since setting the binning
	// size also changes the resolution, and if the resolution is not
	// updated, CUDA will read the old number of pixels determined by
	// the height and width of the image before binning. If these old
	// number of pixels is higher than the current resolution after 
	// setting binning, a segmentation fault will occur as CUDA tries to
	// read memory blocks it does not have access to.
	LOGGER_INFO("Set internal resolution: " << new_config.aoi_index);
	camera_aoi& aoi = aois.at(new_config.aoi_index);


	// set the camera new binning mode. This is for when the
	// binning mode is changed in the GUI
	const camera_bin& new_bin_mode = bin_modes.at(new_config.bin_index);
	const WORD horz_bin{ (WORD)new_bin_mode.s }, vert_bin{ (WORD)new_bin_mode.s };
	PCO_ERR_CHK(PCO_SetBinning(cam_handle, horz_bin, vert_bin));

	/* Cannot set the gain mode as it seems to not be supported by the PCO camera family */

	// set ROI
	auto new_width{ (WORD)aoi.width / horz_bin }, new_height{ (WORD)aoi.height / vert_bin };
	auto x0{1}, y0{1};
	PCO_ERR_CHK(PCO_SetROI(cam_handle, 1, 1, new_width, new_height));

	// send the modified settings to the camera
	PCO_ERR_CHK(PCO_ArmCamera(cam_handle));

	// recompute the size of the images captured by the sensor
	PCO_ERR_CHK(PCO_GetSizes(cam_handle, &XResAct, &YResAct, &XResMax, &YResMax));

	/* re-allocate buffer */
	_buffer.allocate(cam_handle, XResAct, YResAct);


	aoi.re_center_and_fixup(aoi.width, aoi.height);

	// make sure that the camera is healthy
	DWORD CameraWarning, CameraError, CameraStatus;
	PCO_ERR_CHK(PCO_GetCameraHealthStatus(cam_handle, &CameraWarning, &CameraError, &CameraStatus));
	if (CameraError != 0)
	{
		PCO_ERR_CHK(PCO_CloseCamera(cam_handle));
		PCO_ERR_CHK(CameraError);
	}

	// sets the image parameters for internally allocated resources. must be called before image transfer is started
	PCO_ERR_CHK(PCO_SetImageParameters(cam_handle, XResAct, YResAct, IMAGEPARAMETERS_READ_WHILE_RECORDING, 0, 0));
}
#endif