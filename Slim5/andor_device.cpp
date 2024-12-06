#include "stdafx.h"

#if CAMERA_PRESENT_ANDOR == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#pragma comment(lib, "atcorem.lib")
#pragma comment(lib, "atutilitym.lib")
#include "andor_device.h"
#include <iostream>
#include "time_slice.h"
#include <atcore.h>//somehow get rid of this
#include <atutility.h>
#include "qli_runtime_error.h"

//#define track_events
#ifdef track_events
struct AndorEvent
{
	enum class kind { Trigger, Capture, Capture_Failure, SettingChange, ExposureChange, Start, Stop };
	AndorEvent::kind what;
	std::chrono::microseconds when;
	int id;
};
const static std::unordered_map<AndorEvent::kind, std::string> kind_map = {
	{ AndorEvent::kind::Trigger ,"Trigger" },
	{ AndorEvent::kind::Capture ,"Capture" } ,
	{ AndorEvent::kind::Capture_Failure ,"Capture_Failure" },
	{ AndorEvent::kind::SettingChange ,"SettingChange" },
	{ AndorEvent::kind::Start ,"Start" },
	{ AndorEvent::kind::Stop ,"Stop" },
	{ AndorEvent::kind::ExposureChange ,"ExposureChange" }	
};
 static std::map<AndorEvent::kind, int> counter;

class AndorEventHolder
{
	std::vector<AndorEvent> events;
	std::mutex dont_tread_on_snek;
	const static int overflow_size = 128;
public:
	AndorEventHolder() noexcept
	{
		events.reserve(overflow_size);
	}
	void display()
	{
		std::unordered_map<AndorEvent::kind, std::chrono::microseconds> previous_times;
		const auto start = events.begin()->when;
		for (auto andor_event : events)
		{
			std::cout << kind_map.at(andor_event.what) << ", " << to_mili(andor_event.when - start) << "," <<andor_event.id;
			const auto foundIter = previous_times.find(andor_event.what);
			if (foundIter != previous_times.end())
			{
				std::cout << "," << to_mili(andor_event.when - foundIter->second);
			}
			previous_times[andor_event.what] = andor_event.when;
			std::cout << std::endl;
		}
		for (auto& item : counter)
		{
			item.second=0;
		}
		events.resize(0);
		auto here =0;
	}
	void register_event(const AndorEvent::kind kind)
	{
		std::unique_lock<std::mutex> lk(dont_tread_on_snek);
		counter[kind]++;
		events.push_back({ kind,timestamp(),counter[kind]-1 });
		if (kind == AndorEvent::kind::Capture_Failure)
		{
			display();
		}
		if (events.size() == overflow_size)
		{
			events.resize(0);
			//display();
		}
	}
};
AndorEventHolder holder;
#endif

#define ANDOR_SAFE_CALL(err) andor_safe_call(err,__FILE__,__LINE__)
inline void andor_safe_call(const int err, const char* file, const int line)
{
	if (err != AT_SUCCESS)
	{
		std::cout << "Andor failed code: " << err << " @" << line << ":" << file << ": ";
		// specific instructions
		switch (err)
		{
		case AT_ERR_HARDWARE_OVERFLOW: std::cout << "AT_ERR_HARDWARE_OVERFLOW"; break;
		case AT_SUCCESS:std::cout << "AT_SUCCESS"; break;
		case AT_ERR_NOTINITIALISED:std::cout << "AT_ERR_NOTINITIALISED"; break;
		case AT_ERR_NOTIMPLEMENTED:std::cout << "AT_ERR_NOTIMPLEMENTED"; break;
		case AT_ERR_READONLY:std::cout << "AT_ERR_READONLY"; break;
		case AT_ERR_NOTREADABLE:std::cout << "AT_ERR_NOTREADABLE"; break;
		case AT_ERR_NOTWRITABLE:std::cout << "AT_ERR_NOTWRITABLE"; break;
		case AT_ERR_OUTOFRANGE:std::cout << "AT_ERR_OUTOFRANGE"; break;
		case AT_ERR_INDEXNOTAVAILABLE:std::cout << "AT_ERR_INDEXNOTAVAILABLE"; break;
		case AT_ERR_INDEXNOTIMPLEMENTED:std::cout << "AT_ERR_INDEXNOTIMPLEMENTED"; break;
		case AT_ERR_EXCEEDEDMAXSTRINGLENGTH:std::cout << "AT_ERR_EXCEEDEDMAXSTRINGLENGTH"; break;
		case AT_ERR_CONNECTION:std::cout << "AT_ERR_CONNECTION"; break;
		case AT_ERR_NODATA:std::cout << "AT_ERR_NODATA"; break;
		case AT_ERR_INVALIDHANDLE:std::cout << "AT_ERR_INVALIDHANDLE"; break;
		case AT_ERR_TIMEDOUT:std::cout << "AT_ERR_TIMEDOUT"; break;
		case AT_ERR_BUFFERFULL:std::cout << "AT_ERR_BUFFERFU"; break;
		case AT_ERR_INVALIDSIZE:std::cout << "AT_ERR_INVALIDSIZE"; break;
		case AT_ERR_INVALIDALIGNMENT:std::cout << "AT_ERR_INVALIDALIGNMENT"; break;
		case AT_ERR_COMM:std::cout << "AT_ERR_COMM"; break;
		case AT_ERR_STRINGNOTAVAILABLE:std::cout << "AT_ERR_STRINGNOTAVAILABLE"; break;
		case AT_ERR_STRINGNOTIMPLEMENTED:std::cout << "AT_ERR_STRINGNOTIMPLEMENTED"; break;
		case AT_ERR_NULL_FEATURE:std::cout << "AT_ERR_NULL_FEATURE"; break;
		case AT_ERR_NULL_HANDLE:std::cout << "AT_ERR_NULL_HANDLE"; break;
		case AT_ERR_NULL_IMPLEMENTED_VAR:std::cout << "AT_ERR_NULL_IMPLEMENTED_VAR"; break;
		case AT_ERR_NULL_READABLE_VAR:std::cout << "AT_ERR_NULL_READABLE_VAR"; break;
		case AT_ERR_NULL_READONLY_VAR:std::cout << "AT_ERR_NULL_READONLY_VAR"; break;
		case AT_ERR_NULL_WRITABLE_VAR:std::cout << "AT_ERR_NULL_WRITABLE_VAR"; break;
		case AT_ERR_NULL_MINVALUE:std::cout << "AT_ERR_NULL_MINVALUE"; break;
		case AT_ERR_NULL_MAXVALUE:std::cout << "AT_ERR_NULL_MAXVALUE"; break;
		case AT_ERR_NULL_VALUE:std::cout << "AT_ERR_NULL_VALUE"; break;
		case AT_ERR_NULL_STRING:std::cout << "AT_ERR_NULL_STRING"; break;
		case AT_ERR_NULL_COUNT_VAR:std::cout << "AT_ERR_NULL_COUNT_VAR"; break;
		case AT_ERR_NULL_ISAVAILABLE_VAR:std::cout << "AT_ERR_NULL_ISAVAILABLE_VAR"; break;
		case AT_ERR_NULL_MAXSTRINGLENGTH:std::cout << "AT_ERR_NULL_MAXSTRINGLENGTH"; break;
		case AT_ERR_NULL_EVCALLBACK:std::cout << "AT_ERR_NULL_EVCALLBACK"; break;
		case AT_ERR_NULL_QUEUE_PTR:std::cout << "AT_ERR_NULL_QUEUE_PTR"; break;
		case AT_ERR_NULL_WAIT_PTR:std::cout << "AT_ERR_NULL_WAIT_PTR"; break;
		case AT_ERR_NULL_PTRSIZE:std::cout << "AT_ERR_NULL_PTRSIZE"; break;
		case AT_ERR_NOMEMORY:std::cout << "AT_ERR_NOMEMORY"; break;
		case AT_ERR_DEVICEINUSE:std::cout << "AT_ERR_DEVICEINUSE"; break;
		default:
			return;
		}
		std::cout << std::endl;
		qli_runtime_error("Andor Camera Failure, Check Yo Cables!");
	}
}

auto andor_logic_error_msg = "Andor Logic Error";

//static class member because I'm too lazy to solve linking problems
std::chrono::microseconds andor_device::timestamp_delta(AT_U8* p_buf, const size_t buffer_size, const AT_64 clock_rate) noexcept
{
	//following code taken from SDK pdf
#define LENGTH_FIELD_SIZE 4
#define CID_FIELD_SIZE 4
#define TIMESTAMP_FIELD_SIZE 8
	auto* start = p_buf + buffer_size;//end of image
	const auto i_offset = LENGTH_FIELD_SIZE + CID_FIELD_SIZE + TIMESTAMP_FIELD_SIZE;
	start -= i_offset;
	const auto au64_timestamp = *reinterpret_cast<AT_64*>(start);
	//it's made of people!
	const auto human_timestamp = au64_timestamp / (1.0 * clock_rate);
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double, std::ratio<1>>(human_timestamp));
}

AT_H andor_handle;// must be accessible due to windows crap;???

/*------------------Helper Methods---------------*/
void set_enumerated_string(const AT_WC* features, const AT_WC* string)
{
	ANDOR_SAFE_CALL(AT_SetEnumeratedString(andor_handle, features, string));
}

void command(const AT_WC* command)
{
	ANDOR_SAFE_CALL(AT_Command(andor_handle, command));
}

class cool_it : boost::noncopyable  // NOLINT(hicpp-special-member-functions)
{
	//starts cooling while random crap like memory allocation is 'a happening
	const AT_H ath_;
	int temperature_count_, temperature_status_index_;
	std::array<AT_WC, 256> temperature_status_;
	[[nodiscard]] double get_temp() const noexcept
	{
		double temperature;
		ANDOR_SAFE_CALL(AT_GetFloat(ath_, L"SensorTemperature", &temperature));
		return temperature;
	}
	//needs a memory fence of sorts
public:
	explicit cool_it(const AT_H camera) : ath_(camera), temperature_count_(0), temperature_status_index_(0), temperature_status_({ 0 })
	{
		std::cout << "Temperature is " << get_temp() << std::endl;
		{
			ANDOR_SAFE_CALL(AT_SetBool(ath_, L"SensorCooling", AT_TRUE));
			ANDOR_SAFE_CALL(AT_GetEnumCount(ath_, L"TemperatureControl", &temperature_count_));
			std::cout << "Cooling" << std::endl;
		}
	}
	~cool_it()
	{
		{
			constexpr auto patience = 10;
			auto i = 0;
			do
			{
				const auto old_temperature = get_temp();
				windows_sleep(ms_to_chrono(250));
				ANDOR_SAFE_CALL(AT_GetEnumIndex(ath_, L"TemperatureStatus", &temperature_status_index_));
				ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(ath_, L"TemperatureStatus", temperature_status_index_, temperature_status_.data(), 256));
				const auto new_temperature = get_temp();
				std::cout << "Temperature is " << new_temperature << " " << new_temperature - old_temperature << ": " << i << "/" << patience << std::endl;
			} while (wcscmp(L"Stabilized", temperature_status_.data()) != 0 && i++ < patience);
		}
	}
};

struct andor_gain_settings
{
	std::string human_label;
	std::wstring gain_mode_label;
	bool rolling;
	std::wstring speed;
	[[nodiscard]] bool fast_exposure_switching() const noexcept
	{
		return rolling;
	}
	andor_gain_settings(const std::string& label, const std::wstring& gain_mode, const bool is_rolling, const bool is_lower_speed) :
		human_label(label), gain_mode_label(gain_mode), rolling(is_rolling)
	{
		speed = is_lower_speed ? L"100 MHz" : L"280 MHz";
	};
};

std::vector<andor_gain_settings> gain_mode_to_settings = {
	{andor_gain_settings("Rolling", L"16-bit (low noise & high well capacity)" , true,true)},
	{andor_gain_settings("Rolling (faster)", L"16-bit (low noise & high well capacity)" , true,false)},
	{andor_gain_settings("Global", L"16-bit (low noise & high well capacity)" , false,true)},
	{andor_gain_settings("Global (faster)", L"16-bit (low noise & high well capacity)" , false,false) }
};

andor_device::andor_device(const int camera_idx, QObject* parent) : camera_device(camera_device_features(true, true, true, false, camera_contrast_features(camera_chroma::monochrome, demosaic_mode::no_processing, { 70,65535 })), camera_idx, parent), andor_transfer_rate_(ms_to_chrono(0)), clock_rate(0)
{
	//Library + Open Camera
	time_slice t("Andor Init:");
	{
		auto er = 0;
		AT_64 i_number_devices = 0;
		er |= AT_InitialiseLibrary();
		er |= AT_InitialiseUtilityLibrary();//British English :-(
		try
		{
			er |= AT_GetInt(AT_HANDLE_SYSTEM, L"DeviceCount", &i_number_devices);
		}
		catch (...)
		{
			std::cout << "Is another program using the camera?" << std::endl;
		}
		if (i_number_devices <= 0) // RETURNS A NUMBER EVEN WHEN NO HARDWARE DEVICES EXIST!!!
		{
			return;
		}
		const static auto camera_name = 0;
		er |= AT_Open(camera_name, &andor_handle);
		if (er != AT_SUCCESS)
		{
			const auto* msg = "Can't find camera, consider power cycling device";
			qli_runtime_error(msg);
		}
	}
	AT_64 aoi_width_max;
	ANDOR_SAFE_CALL(AT_GetIntMax(andor_handle, L"AOIWidth", &aoi_width_max));
	AT_64 aoi_height_max;
	ANDOR_SAFE_CALL(AT_GetIntMax(andor_handle, L"AOIHeight", &aoi_height_max));

	bin_modes.emplace_back(camera_bin(1));
	bin_modes.emplace_back(camera_bin(2));
	bin_modes.emplace_back(camera_bin(4));
	bin_modes.emplace_back(camera_bin(8));

	aois.emplace_back(camera_aoi(aoi_width_max, aoi_height_max, 0, 0));
	//todo generate with some formula?
	if (aoi_width_max > 2048)
	{
		aois.pop_back();// this size often causes streaming problem so we're removing it
		aois.emplace_back(camera_aoi(2064, 2048, 57, 257));
		aois.emplace_back(camera_aoi(1776, 1760, 201, 401));
		aois.emplace_back(camera_aoi(1920, 1080, 537, 337));
		aois.emplace_back(camera_aoi(1536, 1536, 561, 593));
		aois.emplace_back(camera_aoi(1392, 1040, 561, 593));
		aois.emplace_back(camera_aoi(1500, 256, 953, 530));
		aois.emplace_back(camera_aoi(2560, 256, 953, 1));
		aois.emplace_back(camera_aoi(768, 768, 697, 585));
		aois.emplace_back(camera_aoi(528, 512, 825, 1025));
		aois.emplace_back(camera_aoi(240, 256, 953, 1169));
		aois.emplace_back(camera_aoi(144, 128, 1017, 1217));
	}
	else
	{
		aois.emplace_back(camera_aoi(2048, 2048, 1, 1));
		aois.emplace_back(camera_aoi(1920, 1080, 485, 65));
		aois.emplace_back(camera_aoi(1392, 1040, 505, 329));
		aois.emplace_back(camera_aoi(512, 512, 769, 769));
		aois.emplace_back(camera_aoi(128, 128, 961, 961));
	}
	//extra check
	aois.erase(
		std::remove_if(aois.begin(), aois.end(),
			[&](const camera_aoi& o)
			{
				const auto bad = o.width > aoi_width_max || o.height > aoi_height_max;
				return bad;
			}),
		aois.end());
	// center the ROIs
	for (auto&& aoi : aois)
	{
		aoi.re_center_and_fixup(aoi_width_max, aoi_height_max);
	}
#ifndef _DEBUG
#endif
	//
	common_post_constructor();
}

andor_device::~andor_device()
{
	flush_camera_internal_buffer();
	ANDOR_SAFE_CALL(AT_Close(andor_handle));
	ANDOR_SAFE_CALL(AT_FinaliseUtilityLibrary());
	ANDOR_SAFE_CALL(AT_FinaliseLibrary());
}

void andor_device::trigger_internal()
{
	//this can still silently fail, and there ain't much of a way to account for lost frames...
	ANDOR_SAFE_CALL(AT_Command(andor_handle, L"SoftwareTrigger"));
#ifdef track_events
	holder.register_event(AndorEvent::kind::Trigger);
#endif
}

size_t andor_device::capture_hardware_sequence_internal(const camera_frame_processing_function& process_function, const size_t capture_items, const frame_meta_data_before_acquire& prototype, const channel_settings& channel_settings)
{
	size_t secret_counter = 0;
	const auto cycle_settings = channel_settings.iterator();
	size_t capture_item = 0;
	for (; capture_item < capture_items; ++capture_item)
	{
		for (auto d = 0; d < cycle_settings.cycle_limit.denoise_idx; ++d)
		{
			for (auto p = 0; p < cycle_settings.cycle_limit.pattern_idx; ++p)
			{
				AT_U8* p_buf = nullptr;
				int buffer_out;
				const auto return_code = AT_WaitBuffer(andor_handle, &p_buf, &buffer_out, AT_INFINITE);
				if (return_code != AT_SUCCESS)
				{
					std::cout << "Error code " << return_code << " while acquiring hardware arbitrated sequence" << std::endl;
					goto escape;
				}
				//
				const auto timestamp_chronology = timestamp_delta(p_buf, buffer_out, clock_rate);
				frame_meta_data meta_data_after(prototype, timestamp_chronology);
				meta_data_after.pattern_idx = p;
				const auto expected_size = get_sensor_size(camera_configuration_);
				const auto info = image_info(expected_size, 1, image_info::complex::no);
				const auto frame = camera_frame<unsigned short>(reinterpret_cast<unsigned short*>(p_buf), info, meta_data_after);
				align_and_convert_andor_buffer(process_function, frame);
				//
				secret_counter = secret_counter + 1;
				ANDOR_SAFE_CALL(AT_QueueBuffer(andor_handle, p_buf, buffer_out));
			}
		}
	}
	//
escape:
	ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStop"));
	return capture_item;
}

bool andor_device::capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& exposure, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& process_function)
{
	auto no_failure = true;
	const auto images_to_capture = std::distance(frames.first, frames.second);
	ANDOR_SAFE_CALL(AT_SetInt(andor_handle, L"FrameCount", images_to_capture));
	{
		//Set exposure, then set FPS
		auto exposure_time_seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(exposure).count();
		double min_exposure_time_seconds;
		ANDOR_SAFE_CALL(AT_GetFloatMin(andor_handle, L"ExposureTime", &min_exposure_time_seconds));
		double readout_time;
		ANDOR_SAFE_CALL(AT_GetFloatMin(andor_handle, L"ReadoutTime", &readout_time));
		//

		static size_t in_memory_threshold = 0;//this can be calculated from the other API
		if (in_memory_threshold == 0)
		{
			constexpr static auto str_length = 256;
			wchar_t enum_string[str_length];
			ANDOR_SAFE_CALL(AT_GetString(andor_handle, L"DDR2Type", enum_string, 256));
			auto gbs = 0;
			swscanf_s(enum_string, L"%dGB", &gbs, str_length);
			in_memory_threshold = gbs * 1024ULL * 1024ULL * 1024ULL;
		}
		AT_64 image_size_bytes;

		ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"ImageSizeBytes", &image_size_bytes));
		size_t memory_required = images_to_capture * image_size_bytes;
		const auto get_max_time_in_seconds = [&] {
			if (memory_required > in_memory_threshold)
			{
				double max_interface_transfer_rate;
				ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"MaxInterfaceTransferRate", &max_interface_transfer_rate));
				return  std::max(1 / max_interface_transfer_rate, readout_time);//seconds
			}
			return readout_time;//seconds
		};
		const auto max_transfer_rates_in_s = get_max_time_in_seconds();
		exposure_time_seconds = std::max({ exposure_time_seconds ,min_exposure_time_seconds, max_transfer_rates_in_s });
		ANDOR_SAFE_CALL(AT_SetFloat(andor_handle, L"ExposureTime", exposure_time_seconds));
		double fps;
		ANDOR_SAFE_CALL(AT_GetFloatMax(andor_handle, L"FrameRate", &fps));
		ANDOR_SAFE_CALL(AT_SetFloat(andor_handle, L"FrameRate", fps));
		const auto exposure_time_seconds_chrono = std::chrono::duration<double, std::ratio<1>>(exposure_time_seconds);
		exposure_ = std::chrono::duration_cast<std::chrono::microseconds>(exposure_time_seconds_chrono);
	}
	//ANDOR_SAFE_CALL(AT_Command(andor_handle, L"TimestampClockReset"));
	ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStart"));
	for (auto it = frames.first; it < frames.second; ++it)
	{
		AT_U8* p_buf = nullptr;
		auto buffer_out = 0;
		const auto timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_time_out).count();
		const auto return_code = AT_WaitBuffer(andor_handle, &p_buf, &buffer_out, timeout_ms);
		if (return_code == AT_SUCCESS)
		{
			const auto timestamp_chrono = timestamp_delta(p_buf, buffer_out, clock_rate);
			frame_meta_data meta_data_after(meta_data, timestamp_chrono);
			meta_data_after.exposure_time = this->exposure_;
			const auto expected_bytes = static_cast<int>(this->get_sensor_bytes(camera_configuration_));
			if (expected_bytes > buffer_out)
			{
				qli_runtime_error(andor_logic_error_msg);
			}
			const auto expected_size = get_sensor_size(camera_configuration_);
			const auto info = image_info(expected_size, 1, image_info::complex::no);
			auto frame = camera_frame<unsigned short>(reinterpret_cast<unsigned short*>(p_buf), info, meta_data_after);
			if (!frame.is_valid())
			{
				qli_runtime_error(andor_logic_error_msg);
			}
			{
				align_and_convert_andor_buffer(process_function, frame);
			}
			ANDOR_SAFE_CALL(AT_QueueBuffer(andor_handle, p_buf, buffer_out));
		}
		else
		{
			std::cout << "Andor Error " << return_code << std::endl;
			if (return_code == AT_ERR_TIMEDOUT)
			{
				const auto acquired = std::distance(frames.first, frames.second);
				std::cout << "Camera buffer overflow" << return_code << ", acquired " << acquired << " frames before failing" << std::endl;
			}
			no_failure = false;
			goto escape;
		}
	}
escape:
	ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStop"));
	return no_failure;
}

bool andor_device::capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout)
{
	//auto what = timestamp();
	auto* p_buf = static_cast<AT_U8*>(nullptr);
	auto buf_size = 0;
	//check if these have the right units
	//const unsigned int wait_time_final = round(to_mili(ms_to_chrono(1000)));
	const unsigned int wait_time_final = round(to_mili(timeout)*3);
	int result;
	{
		result = AT_WaitBuffer(andor_handle, &p_buf, &buf_size, wait_time_final);
	}
	const auto ptr_check = inside_camera_.front() == p_buf;
	const auto success = result == AT_SUCCESS && ptr_check;
#ifdef track_events
	holder.register_event(success ? AndorEvent::kind::Capture : AndorEvent::kind::Capture_Failure);
#endif	
	if (!success)
	{
		// std::cout << "Acquisition Error: " << result << ":" << std::endl;
	}
	else
	{
		inside_camera_.pop();
		const auto timestamp_chrono = timestamp_delta(p_buf, buf_size, clock_rate);
		const frame_meta_data meta_data_after(meta_data, timestamp_chrono);
		const auto camera_config = get_camera_config();
		const auto expected_frame_size = get_sensor_size(camera_config);
		const auto expected_size_bytes = get_sensor_bytes(camera_config);
		if (expected_size_bytes > buf_size)
		{
			qli_runtime_error(andor_logic_error_msg);
		}
		const auto info = image_info(expected_frame_size, 1, image_info::complex::no);
		const auto frame = camera_frame<unsigned short>(reinterpret_cast<unsigned short*>(p_buf), info, meta_data_after);
		if (!frame.is_valid())
		{
			qli_runtime_error(andor_logic_error_msg);
		}

		align_and_convert_andor_buffer(fill_me, frame);
		ANDOR_SAFE_CALL(AT_QueueBuffer(andor_handle, p_buf, buf_size));
		inside_camera_.push(p_buf);
	}
#ifdef track_events
	holder.register_event(AndorEvent::kind::Capture);
#endif
	return success;
}

void andor_device::fix_camera_internal()
{
	flush_camera_internal_buffer();
	allocate_memory_pool();
}

#define EXTRACTLOWPACKED(SourcePtr) ( ((SourcePtr)[0] << 4) + ((SourcePtr)[1] & 0xF) )
#define EXTRACTHIGHPACKED(SourcePtr) ( ((SourcePtr)[2] << 4) + ((SourcePtr)[1] >> 4) )
double pixel_readout_time()
{
	double readout_time;
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"ReadoutTime", &readout_time));
	return 1000 * (readout_time + 0.00015);//15 is longest time for row
}

void andor_device::apply_settings_internal(const camera_config& new_config)
{
	auto& gain_info = gain_mode_to_settings.at(new_config.gain_index);
	const auto bin_num = new_config.bin_index;
	const auto size_num = new_config.aoi_index;
	//Note, the Andor manual says these functions should be called in a certain order
	//////Shuttering
	const auto* rolling = gain_info.rolling ? L"Rolling" : L"Global";
	ANDOR_SAFE_CALL(AT_SetEnumeratedString(andor_handle, L"ElectronicShutteringMode", rolling));
	//////Gain
	const auto get_as_w_hack = [](const std::string& s)
	{
		std::wstring ws(s.size(), L' '); // Overestimate number of code points.
		// ReSharper disable once CppDeprecatedEntity
		ws.resize(std::mbstowcs(&ws[0], s.c_str(), s.size())); // Shrink to fit.;
		return ws;
	};
	ANDOR_SAFE_CALL(AT_SetEnumeratedString(andor_handle, L"SimplePreAmpGainControl", gain_info.gain_mode_label.c_str()));
	//////Noise
	ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"SpuriousNoiseFilter", AT_TRUE));//Might be fun to investigate this
	ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"StaticBlemishCorrection", AT_TRUE));
	//////ReadoutRate
	ANDOR_SAFE_CALL(AT_SetEnumeratedString(andor_handle, L"PixelReadoutRate", gain_info.speed.c_str()));
	//////TriggerMode
	const std::unordered_map<camera_mode, std::wstring> trigger_modes = { { camera_mode::software, L"Software" },{ camera_mode::burst, L"Internal" } ,{ camera_mode::hardware_trigger, L"External Exposure" } };
	ANDOR_SAFE_CALL(AT_SetEnumeratedString(andor_handle, L"TriggerMode", trigger_modes.at(new_config.mode).c_str()));
	//////OverlapMode
	if (new_config.mode == camera_mode::burst)
	{
		ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"Overlap", AT_TRUE));
	}
	if (new_config.mode == camera_mode::hardware_trigger)
	{
		ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"Overlap", AT_FALSE));
	}
	//////CycleMode
	{
		const auto* cycle_mode = new_config.mode == camera_mode::burst ? L"Fixed" : L"Continuous";
		ANDOR_SAFE_CALL(AT_SetEnumString(andor_handle, L"CycleMode", cycle_mode));
	}
	//////ImageSize
	const auto aoi = aois.at(size_num);
	auto& bin_mode = bin_modes.at(bin_num);
	const auto bin_label = bin_mode.to_string().toStdString();
	const auto roi_bin = get_as_w_hack(bin_label);
	const auto camera_check = AT_SetEnumeratedString(andor_handle, L"AOIBinning", roi_bin.c_str()) != AT_SUCCESS;
	if (camera_check)
	{
		const auto* msg = "Can't find camera, consider power cycling device";
		qli_runtime_error(msg);
	}
	const auto scale = bin_mode.s;
	if (aoi.width % scale != 0 || aoi.height % scale != 0)
	{
		qli_runtime_error();
	}
	ANDOR_SAFE_CALL(AT_SetInt(andor_handle, L"AOIWidth", aoi.width / scale));//Division might not cast to right size, check if these need to be ceiled or florred
	ANDOR_SAFE_CALL(AT_SetInt(andor_handle, L"AOIHeight", aoi.height / scale));
	ANDOR_SAFE_CALL(AT_SetInt(andor_handle, L"AOILeft", 1 + aoi.left));
	AT_BOOL roi_centering_implemented;
	ANDOR_SAFE_CALL(AT_IsImplemented(andor_handle, L"VerticallyCentreAOI", &roi_centering_implemented));
	if (roi_centering_implemented)
	{
		ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"VerticallyCentreAOI", AT_TRUE));
	}
	//
	ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"MetadataEnable", true));
	ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"MetadataTimestamp", true));
	//
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"TimestampClockFrequency", &clock_rate));
	//
	andor_transfer_rate_ = get_min_copy_back_in_mili();
#ifdef track_events
	print_debug(std::cout);
	holder.register_event(AndorEvent::kind::SettingChange);
#endif
	//
	allocate_memory_pool();
}

void andor_device::set_exposure_internal(const std::chrono::microseconds& exposure)
{
	//////Keep track of old state
	const auto seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(exposure);
	const auto seconds_double = seconds.count();
	const auto can_set_exposure = gain_mode_to_settings.at(camera_configuration_.gain_index).fast_exposure_switching();
	if (can_set_exposure)
	{
		ANDOR_SAFE_CALL(AT_SetFloat(andor_handle, L"ExposureTime", seconds_double));
	}
	else
	{
		const auto should_stop = [&]
		{
			AT_BOOL acquiring;
			ANDOR_SAFE_CALL(AT_GetBool(andor_handle, L"CameraAcquiring", &acquiring));
			return (acquiring == AT_TRUE);
		}();
		if (should_stop)
		{
			ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStop"));
		}
		ANDOR_SAFE_CALL(AT_SetFloat(andor_handle, L"ExposureTime", seconds_double));
		if (should_stop)
		{
			ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStart"));
		}
	}
#ifdef track_events
	holder.register_event(AndorEvent::kind::ExposureChange);
#endif
}

void andor_device::print_debug(std::ostream& input)
{
	input << std::endl;
	input << "Camera Settings" << std::endl;
	double temp=42.2;
	int temp_in=0;
	AT_64 temp_at64 =12;
	constexpr auto n = 128;
	AT_WC temp_text[n];
	char text_output[n];
	AT_BOOL temp_bool;
	//AOI
	AT_64 left, top, height, width;
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"AOILeft", &left));
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"AOITop", &top));
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"AOIHeight", &height));
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"AOIWidth", &width));
	input << "AOI set to " << width << "," << height << "," << top << "," << left << std::endl;
	//AOI Stride
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"AOIStride", &temp_at64));
	input << "AOI stride " << temp_at64 << std::endl;
	//Binning
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"AOIBinning", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"AOIBinning", temp_in, temp_text, n));
	size_t i;
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "AOIBinning : " << text_output << std::endl;
	//TemperatureStatus
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"TemperatureStatus", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"TemperatureStatus", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "TemperatureStatus : " << text_output << std::endl;
	//CycleMode
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"CycleMode", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"CycleMode", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "CycleMode : " << text_output << std::endl;
	//ElectronicShutteringMode
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"ElectronicShutteringMode", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"ElectronicShutteringMode", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "ElectronicShutteringMode : " << text_output << std::endl;
	//pixelreadout
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"PixelReadoutRate", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"PixelReadoutRate", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "PixelReadoutRate : " << text_output << std::endl;
	//SimplePreAmpGainControl
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"SimplePreAmpGainControl", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"SimplePreAmpGainControl", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "SimplePreAmpGainControl : " << text_output << std::endl;
	//PreAmpGain
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"PreAmpGain", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"PreAmpGain", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "PreAmpGain : " << text_output << std::endl;
	//PreAmpGainChannel
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"PreAmpGainChannel", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"PreAmpGainChannel", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "PreAmpGainChannel : " << text_output << std::endl;
#ifdef doNeo //PreAmpGainControl deprecated on Zyla
	andorSafeCall(AT_GetEnumIndex(andorHandle, L"PreAmpGainControl", &tempIn));
	andorSafeCall(AT_GetEnumStringByIndex(andorHandle, L"PreAmpGainControl", tempIn, tempText, 64));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "PreAmpGainControl : " << textOutput << endl;
	//PreAmpGainSelector
	andorSafeCall(AT_GetEnumIndex(andorHandle, "PreAmpGainSelector", &tempIn));
	andorSafeCall(AT_GetEnumStringByIndex(andorHandle, "PreAmpGainSelector", tempIn, tempText, 64));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "PreAmpGainSelector : " << textOutput << endl;
#endif
	//Overlap
	ANDOR_SAFE_CALL(AT_GetBool(andor_handle, L"Overlap", &temp_bool));
	input << "Overlap : " << static_cast<int>(temp_bool) << std::endl;
	//AOI Control
	ANDOR_SAFE_CALL(AT_GetBool(andor_handle, L"FullAOIControl", &temp_bool));
	input << "FullAOIControl : " << static_cast<int>(temp_bool) << std::endl;
	//SpuriousNoiseFilter
	ANDOR_SAFE_CALL(AT_GetBool(andor_handle, L"SpuriousNoiseFilter", &temp_bool));
	input << "SpuriousNoiseFilter : " << static_cast<int>(temp_bool) << std::endl;
	//TriggerMode
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"TriggerMode", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"TriggerMode", temp_in, temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "TriggerMode : " << text_output << std::endl;
	//ExposureTime
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"ExposureTime", &temp));
	input << "ExposureTime Requested:  " << 1000 * temp << std::endl;
	//ReadoutTime
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"ReadoutTime", &temp));
	input << "ReadoutTime Requested:  " << 1000 * temp << std::endl;
	//ActualExposureTime
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"ActualExposureTime", &temp));
	input << "ExposureTime Actual:  " << 1000 * temp << std::endl;
	//MaxInterfaceTransferRate
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"MaxInterfaceTransferRate", &temp));
	input << "MaxInterfaceTransferRate:  " << temp << std::endl;
	//BaselineLevel
	//andorSafeCall(AT_GetFloat(andorHandle, "BaselineLevel", &temp));
	//cout << "BaselineLevel:  " <<temp << endl;
	//assert(temp < 1);//hardcoded so we don't break our camera! (also appears not to be hardcoded but rather commented
	//FrameRate
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"FrameRate", &temp));
	input << "Max Framerate: " << temp << " fps, or " << 1000.0f / temp << " ms" << std::endl;
	//ReadoutTime
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"ReadoutTime", &temp));
	input << "Readout Time: " << temp * 1000 << std::endl;
	//ImageSize
	//PixelEncoding
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"PixelEncoding", &temp_in));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"PixelEncoding", temp_in, temp_text, 64));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "PixelEncoding: " << text_output << std::endl;
	//BytesPerPixel
	//InterfaceType
	ANDOR_SAFE_CALL(AT_GetString(andor_handle, L"InterfaceType", temp_text, n));
	wcstombs_s(&i, text_output, n * sizeof(char), temp_text, _TRUNCATE);
	input << "InterfaceType: " << text_output << std::endl;
	//
	input << std::endl;
}

void andor_device::start_capture_internal()
{
	//will start and wait for trigger there is a 150 ms warmup time. during which the CPU can do useful things
	AT_BOOL acquiring;
	ANDOR_SAFE_CALL(AT_GetBool(andor_handle, L"CameraAcquiring", &acquiring));
	if (acquiring != AT_FALSE)
	{
		qli_runtime_error("");
	}
	//flushes the pool
	allocate_memory_pool();//typically doesn't matter unless the settings changed...
	//ANDOR_SAFE_CALL(AT_Command(andor_handle, L"TimestampClockReset"));
	ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStart"));
	//
	windows_sleep(ms_to_chrono(1000));
#ifdef track_events
	holder.register_event(AndorEvent::kind::Start);
#endif
}

void andor_device::stop_capture_internal()
{
	AT_BOOL acquiring;
	ANDOR_SAFE_CALL(AT_GetBool(andor_handle, L"CameraAcquiring", &acquiring));
	if (acquiring == AT_TRUE)//strange destructor order
	{
		ANDOR_SAFE_CALL(AT_Command(andor_handle, L"AcquisitionStop"));// check if it should be flushed?
	}
#ifdef track_events
	holder.register_event(AndorEvent::kind::Stop);
#endif	
}

std::chrono::microseconds andor_device::get_min_exposure_internal()
{
	double value_double;
	ANDOR_SAFE_CALL(AT_GetFloatMin(andor_handle, L"ExposureTime", &value_double));//in seconds
	const auto time = std::chrono::duration<double, std::ratio<1>>(value_double);
	const auto value = std::chrono::duration_cast<std::chrono::microseconds>(time);
	return value;
}

std::chrono::microseconds andor_device::get_readout_time_internal()
{
	double readout_time;
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"ReadoutTime", &readout_time));//in seconds
	const auto time = std::chrono::duration<double, std::ratio<1>>(readout_time);
	const auto value = std::chrono::duration_cast<std::chrono::microseconds>(time);
	return value;
}

std::chrono::microseconds andor_device::get_transfer_time_internal()
{
	const auto min_readout = get_readout_time_internal();
	const auto min_copy_back = get_min_copy_back_in_mili();
	const auto min_frame_rate = get_min_fps_in_mili();
	const auto max_time = std::max({ min_readout, min_copy_back, min_frame_rate });
	return max_time;//lets add 1 for jitter?
}

std::chrono::microseconds andor_device::get_min_fps_in_mili() const
{
	double temp;
	ANDOR_SAFE_CALL(AT_GetFloatMax(andor_handle, L"FrameRate", &temp));
	const auto cycle_time_seconds = 1 / temp;
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double, std::ratio<1>>(cycle_time_seconds));
}

std::chrono::microseconds andor_device::get_min_copy_back_in_mili() const
{
	double max_interface_transfer_rate;
	//andorSafeCall(AT_GetFloat(andorHandle, "FrameRate", &temp));
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"MaxInterfaceTransferRate", &max_interface_transfer_rate));
	const auto cycle_time_seconds = 1 / max_interface_transfer_rate;
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double, std::ratio<1>>(cycle_time_seconds));
}

void andor_device::flush_camera_internal_buffer()
{
	ANDOR_SAFE_CALL(AT_Flush(andor_handle));
	inside_camera_ = std::queue<unsigned char*>();
}

void andor_device::allocate_memory_pool()
{
	AT_64 image_size_bytes;
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"ImageSizeBytes", &image_size_bytes));
	if (!inside_camera_.empty())
	{
		qli_runtime_error(andor_logic_error_msg);
	}
	camera_buffer_.resize(andor_internal_pool_count);
	for (auto i = 0; i < andor_internal_pool_count; i++)
	{
		auto& camera_buffer = camera_buffer_.at(i);
		camera_buffer.resize(image_size_bytes);
		auto* ptr = camera_buffer.data();
		inside_camera_.push(ptr);
		ANDOR_SAFE_CALL(AT_QueueBuffer(andor_handle, ptr, image_size_bytes));
	}
}

void  andor_device::set_cooling_internal(const bool enable)
{
	const auto status = enable ? AT_TRUE : AT_FALSE;
	ANDOR_SAFE_CALL(AT_SetBool(andor_handle, L"SensorCooling", status));
	const std::unordered_map<bool, std::wstring> trigger_modes = { { false, L"Off" },{ true, L"On" } };
	ANDOR_SAFE_CALL(AT_SetEnumString(andor_handle, L"FanSpeed", trigger_modes.at(enable).c_str()));
	int temperature_status_index;
	static std::array<AT_WC, 256> temperature_status;
	ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"TemperatureStatus", &temperature_status_index));
	ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"TemperatureStatus", temperature_status_index, temperature_status.data(), temperature_status.size()));
	double temperature;
	ANDOR_SAFE_CALL(AT_GetFloat(andor_handle, L"SensorTemperature", &temperature));
	std::cout << (enable ? "Enabling" : "Disabling") << " cooling. Current temperature is " << temperature;
	std::wcout << L"(c). Device status is '" << temperature_status.data() << std::endl;
}

void andor_device::align_and_convert_andor_buffer(const camera_frame_processing_function& fill_me, const camera_frame<unsigned short>& ptr_raw)
{
	AT_64 buffer_row_size;
	ANDOR_SAFE_CALL(AT_GetInt(andor_handle, L"AOIStride", &buffer_row_size));
	if (buffer_row_size == ptr_raw.width * sizeof(unsigned short))
	{
		fill_me(ptr_raw);
	}
	else
	{
		//maybe get from metadata? (this might not work on all models...)
		auto munged = ptr_raw;
		auto* ptr_in = reinterpret_cast<AT_U8*>(ptr_raw.img);
		bit_convert_buffer_.resize(ptr_raw.n() * sizeof(unsigned short));
		munged.img = reinterpret_cast<unsigned short*>(bit_convert_buffer_.data());
		auto pixel_encoding_idx=0;
		const static auto encoding_length = 64;
		AT_WC encoding[encoding_length];
		ANDOR_SAFE_CALL(AT_GetEnumIndex(andor_handle, L"PixelEncoding", &pixel_encoding_idx));
		ANDOR_SAFE_CALL(AT_GetEnumStringByIndex(andor_handle, L"PixelEncoding", pixel_encoding_idx, encoding, encoding_length));//call onece?
		ANDOR_SAFE_CALL(AT_ConvertBuffer(ptr_in, reinterpret_cast<AT_U8*>(munged.img), ptr_raw.width, ptr_raw.height, buffer_row_size, encoding, L"Mono16"));
		munged.exposure_time = exposure_;
		fill_me(munged);
	}
}

QStringList andor_device::get_gain_names_internal() const
{
	QStringList names;
	for (auto& item : gain_mode_to_settings)
	{
		auto label = QString::fromStdString(item.human_label);
		names.push_back(label);
	}
	return names;
}


#endif