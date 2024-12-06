#pragma once
#ifndef pco_
#define pco_

#include "stdafx.h"
#include <iostream>
#include <camera_device.h>
#include <qli_runtime_error.h>

// include camera specific header files. These should be outlined
// in the SDK manual
#include <PCO_err.h>
#include <PCO_errtext.h>
#include <sc2_defs.h>
#include <SC2_SDKAddendum.h>
#include <sc2_SDKStructures.h>
#include <sc2_common.h>
#include <SC2_CamExport.h>

using std::cout;
using std::endl;

/**
 * @brief Wrapper for all PCO API calls. This is to check the error code returned by any PCO
 *      API function call and either display a warning message on the command prompt or to
 *      exit the program by throwing a runtime error in the instance of an error.
 */
#ifndef PCO_ERR_CHK
#define PCO_ERR_CHK(error_code) __pco_err_chk(error_code, __FILE__, __LINE__)
#endif

 /**
  * @brief Function for checking the error code returned by any PCO API function call. If the
  *      error bit in the error code is set, then a runtime error is thrown with the code and
  *      the message associated with that particular code. If the warning bit is set, a
  *      message is printed on the command prompt containing the hexadecimal warning code, and
  *      the message associated with that code.
  *
	 <table>
	 <caption id="multi_row">Description of bits in error code</caption>
	 <tr>
	 <th>Bits
	 <th>Description
	 <tr>
	 <td>0 - 11
	 <td>Used to indicate the error or warning number. This is the index
									 of the message in the error or warning C-style string array.
	 <tr>
	 <td>12 - 15
	 <td>Shows the layer code of the error source. This indicates which
									 error or warning C-style string array to reference.
	 <tr>
	 <td>16 - 23
	 <td>Reflects the error source. From the <em>PCO_errtext.h</em> it
									 seems that these are all zeros, and are not used.
	 <tr>
	 <td>24 - 28
	 <td>These bits are not used.
	 <tr>
	 <td>29
	 <td>The common error group flag. This flag is used to lookup the
									 error text inside the correct array. <em><strong>This is not
									 used in this function.</strong></em>
	 <tr>
	 <td>30
	 <td>If this bit is set, it indicates an error.
	 <tr>
	 <td>31
	 <td>If this bit is set it indicates a warning. If the warning bit is
									 set, almost always the error bit is also set.
	 </table>
  *
  * @param err_code signed 32 bit integer code returned by any PCO API function call
  * @param file The file in which this function is called
  * @param line The line at which this function is called
 */
void __pco_err_chk(const int err_code, const char* file, const int line);

class pco : public camera_device
{
protected:

	/*!
	 *  Struct for storing PCO buffer related management.
	 */
	typedef class camera_internal_buffer {
	public:
		/*!
		 *  Default Constructor. Initializes this class with 16 buffers.
		 *  According to the PCO SDK, 16 is the maximum number of buffers
		 *  that cam be allocated internally in the camera.
		 */
		camera_internal_buffer() : camera_internal_buffer(16) {}

		/*!
		 *  Construct the camera internal buffer object.
		 *
		 *      @param [in] num_buffers Specifies the number of camera buffers to create.
		 *      According to the PCO SDK manual, these cameras can have at most 16 buffers.
		 */
		explicit camera_internal_buffer(size_t num_buffers) : _num_buffers(num_buffers) {
			qli_assert(num_buffers < 17, "number of buffers cannot be more than 16");

			BufEvent = new HANDLE[num_buffers];
			BufNum = new short[num_buffers]();
			BufAdr = new WORD * [num_buffers]();

			for (size_t i = 0; i < _num_buffers; ++i) {
				BufNum[i] = -1;
				BufAdr[i] = (WORD*)0;
			}
		}

		/*!
		 *  Destructor. Free the allocate memory. Note that this does
		 *  not free the memory allcoated in the camera internals. That
		 *  needs to be freed manually by calling PCO_FreeBuffer.
		 */
		~camera_internal_buffer() {
			if (BufEvent) {
				delete[] BufEvent;
				BufEvent = (HANDLE*)0;
			}

			if (BufNum) {
				delete[] BufNum;
				BufNum = (SHORT*)0;
			}

			if (BufAdr) {
				delete[] BufAdr;
				BufAdr = (WORD**)0;
			}
		}

		/*!
		 *  Frees the allocated buffers in camera by using calls to PCO_FreeBuffer.
		 *
		 *      @param [in] cam_handle Handle to a previously opened camera device
		 */
		void free_buffers_in_camera(const HANDLE& cam_handle) const;

		/*!
		 *  Allocates the camera internal buffer.
		 *
		 *      @param [in,out] cam_handle
		 *      @param [in]     xResAct
		 *      @param [in]     yResAct
		 */
		void allocate(HANDLE& cam_handle, size_t xResAct, size_t yResAct);

		/*!
		 *  The number of buffers allocated in the camera internally.
		 */
		size_t _num_buffers;

		/*!
		 *  The buffer event handler (one for each buffer)
		 */
		HANDLE* BufEvent;

		/*!
		 *  Since there can be at most 16 buffers. This variable represents
		 *  a pointer to an array of buffer numbers. The maximum size of this
		 *  array can be 16. Each element of this array can be one of -1 or
		 *  0...15. The initial value is -1. This asks calls to PCO_AllocateBuffer
		 *  to create a new buffer. A value between 0 an 15 inclusive, indicates
		 *  a buffer index from a previous PCO_AllocateBuffer call.
		 */
		short* BufNum;

		/*!
		 *  An array of addresses to allocated internal buffers. This points
		 *  to the actual buffers allocated in the camera.
		 */
		WORD** BufAdr;

		/*!
		 *  Used to get the wait status when using PCO_AddBufferEx or any PCO
		 *  API family of functions that requires waiting on a camera handle
		 */
		DWORD waitstat;
	} _buffer_t;


	/**
	 * @brief Current set image width
	*/
	WORD XResAct;

	/**
	 * @brief Current set image height
	*/
	WORD YResAct;

	/**
	 * @brief Maximum image width
	*/
	WORD XResMax;

	/**
	 * @brief Maximum image height
	*/
	WORD YResMax;

	/*!
	 *  Variable for accessing PCO buffer related variables
	 *  such as buffer event handler, buffer number, buffer
	 *  address, number of buffers, etc.
	 */
	_buffer_t _buffer;

	/*!
	 *  Initializes the (Area of Interests) aois inherited vector from camera_device
	 *  class to the available regions of interest (ROIs) supported by the
	 *  PCO camera connected to the system. For this specific version
	 *  we want the ROIs for PCO Panda 4.2. Refer to section A1.1 of
	*   <a href="https://www.pco.de/fileadmin/fileadmin/user_upload/pco-manuals/pco.panda_manual.pdf">
	*   pco.panda manual
	*  </a>
	 */
	virtual void initialize_aois() = 0;

	/*!
	 *  Adds the bin modes to the bin_modes vector inherited from camera_device class.
	 *  For this particular class we want the bin modes associated with PCO Panda 4.2
	 *  camera version. These can be found in the PCO Panda 4.2 users manual. Refer to
	 *  section A1.1 of
	 *  <a href="https://www.pco.de/fileadmin/fileadmin/user_upload/pco-manuals/pco.panda_manual.pdf">
	 *  pco.panda manual. Different PCO cameras can have different bin modes.
	 *  </a>
	 */
	void add_bin_modes();

	/*!
	 *  Fills the pco descriptors. In this context, PCO descriptors are just PCO camera structures.
	 *  These structures are defined in both the manual and sc2_structures.h file.
	 */
	void fill_pco_descriptors();

	/*!
	 *  Initialized the PCO camera.
	 */
	void init();

public:

	HANDLE cam_handle;
	PCO_General strGeneral;
	PCO_CameraType strCamType;
	PCO_Sensor strSensor;
	PCO_Description strDescription;
	PCO_Timing strTiming;
	PCO_Storage strStorage;
	PCO_Recording strRecording;
	WORD RecordingState;

	/**
		* @brief Constructor. Set all the buffer size parameters tthe expected values.
		*      Open the camera link and fill the SDK structures.
		* @param camera_id In the event that there are multiple cameras connected tthe
		*      system, this integer variable uniquely identifies each camera.
		* @param parent
	*/
	explicit pco(const camera_device_features& cdf, size_t buf_size, int camera_id, QObject* parent = NULL);

	/**
		* @brief Destructor. When this is called, all the allocated memory using the **new**
		*      is deleted and the camera handle structure is closed
	*/
	virtual ~pco();

	/*!
	 *  Flushes the camera internal buffer. This function does not need to be implemented in
	 *  the base class. If any concrete camera class needs it, they are free to do so. As
	 *  implementing this function in the PCO base class might break future PCO camera
	 *  integrations
	 */
	void flush_camera_internal_buffer() override;

	/*!
	 *  Implement the function in this class, as it seems none of the PCO camera
	 *  family have gain
	 */
	[[nodiscard]] QStringList get_gain_names_internal() const override;

	/**
	 * @brief Refer to section 2.6.4 and 2.6.5 of the PCO SDK manual
	 * @param exposure The new microsecond exposure time
	*/
	void set_exposure_internal(const std::chrono::microseconds& exposure)  override;

	/*!
	 *  Returns the pco's internal buffer count. This is the number of buffers allocated
	 *
	 *      @return The number of buffers allocated.
	 */
	[[nodiscard]] __forceinline int get_internal_buffer_count() const override
	{
		return _buffer._num_buffers;
	}

	void print_debug(std::ostream& input)  override;

	/*!
	 *  Returns the pco's transfer time for transfering images to PC.
	 *
	 *      @return The transfer time in microsecond.
	 */
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override;

	/*!
	 *  Returns the pco's min exposure.
	 *
	 *      @return The min exposure time of the camera.
	 */
	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;

	/*!
	 *  Returns the pco's frame readout time.
	 *
	 *      @return The time it takes to read per frame.
	 */
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;

	/*!
	 *  Captures the burst internal. This is only supported for GigE connections, and thus
	 *  is only supported in a few PCO cameras
	 *
	 *      @param [in] frames
	 *      @param [in] meta_data
	 *      @param [in] exposure
	 *      @param [in] frame_time_out
	 *      @param [in] process_function
	 *
	 *      @return
	 */
	bool capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator,
		std::vector<capture_item>::const_iterator>& frames,
		const frame_meta_data_before_acquire& meta_data,
		const std::chrono::microseconds& exposure,
		const std::chrono::microseconds& frame_time_out,
		const camera_frame_processing_function& process_function)  override;

	/*!
	 *  Starts the capture internal. Prepares the pco camera for recording and
	 *  transferring images
	 */
	void start_capture_internal()  override;

	/*!
	 *  Stops the capture internal. Stops the camera from recording images in its
	 *  internal buffer
	 */
	void stop_capture_internal()  override;

	/*!
	 *  Publishes the frame. Publish the captured frame to the rest of the program.
	 */
	virtual void publish_frame(const camera_frame_processing_function& fill_me,
		const frame_meta_data_before_acquire& meta_data,
		const uint32_t& index) const
	{
		// pass the data to the rest of the program
		const std::chrono::microseconds capture_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::system_clock::now().time_since_epoch()
			); // apparently I could not find how to get the image timestamp out of the PCO panda camera
		frame_meta_data meta_data_after_acquire(meta_data, capture_timestamp);
		const camera_aoi& roi = aois[camera_configuration_.aoi_index];
		const image_info info = image_info(roi.to_frame_size(), 1, image_info::complex::no);
		camera_frame<WORD> frame(_buffer.BufAdr[index], info, meta_data_after_acquire);
		fill_me(frame);
	}

	/*!
	 *  This function is not needed for PCO Pands 4.2 as it is a passively cooled camera.
	 *  By default this is always not implemented since not all PCO cameras support active
	 *  cooling.
	 */
	void set_cooling_internal(bool enable) override;

	/*!
	 *  Fixes the camera internal if an error has occurred when capturing from the camera.
	 *  This function reinitializes all the camera internals, and re-arms the camera.
	 */
	void fix_camera_internal() override;

	/*!
	 *  This function is not used in PCO Panda 4.2. Apparently with this camera, when a
	 *  software trigger flag is set the camera gives a driver error.
	 */
	void trigger_internal() override;

	/*!
	 *  Captures the internal.
	 *
	 *      @param [in] fill_me
	 *      @param [in] meta_data
	 *      @param [in] timeout
	 *
	 *      @return
	 */
	bool capture_internal(const camera_frame_processing_function& fill_me,
		const frame_meta_data_before_acquire& meta_data,
		const std::chrono::microseconds& timeout) override;

	/**
	 * @brief
	 *
	 *      @param [in] new_config The new camera setting to be applied. This is usually
	 *      passed from the GUI.
	*/
	void apply_settings_internal(const camera_config& new_config) override;
};

#endif
