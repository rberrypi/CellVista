#include "stdafx.h"
#include "virtual_camera_device.h"
#include "device_factory.h"
#include "slm_device.h"
#include <QResource>
#include <iostream>
#include <QMessageBox>
#include <sstream>

#include "qli_runtime_error.h"
#include "virtual_camera_settings.h"

[[nodiscard]] int virtual_camera_device::get_internal_buffer_count() const
{
	return 30;
}

void virtual_camera_device::flush_camera_internal_buffer()
{
	//no need to implement
}

QStringList virtual_camera_device::get_gain_names_internal() const
{
	QStringList in_modes;
	in_modes << "none";
	return in_modes;
}

void virtual_camera_device::trigger_internal()
{
	windows_sleep(exposure_);
	{
		std::unique_lock<std::mutex> lk(frame_queue_simulation_m_);
		const auto pattern = D->get_slm_frame_idx();
		const acquisition_simulation virtual_frame = { timestamp(), pattern };
		frame_queue_simulation_.push(virtual_frame);
	}
	frame_queue_simulation_cv_.notify_one();
}

bool virtual_camera_device::capture_burst_internal(const std::pair<capture_item_iterator, capture_item_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& exposure, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& processing_function)
{
	//trigger them all, note PSI is disabled during this phase
	{
		for (auto it = frames.first; it < frames.second; ++it)
		{
			windows_sleep(exposure);
			trigger_internal();
		}
	}
	// read them all back
	{
		for (auto it = frames.first; it < frames.second; ++it)
		{
			capture_internal(processing_function, meta_data, frame_time_out);
		}
	}
	return true;
}

bool virtual_camera_device::capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& frame_meta_data_before_acquire, const std::chrono::microseconds& timeout)
{
	std::unique_lock<std::mutex> lk(frame_queue_simulation_m_);
	const auto has_item = [&]
	{
		return !frame_queue_simulation_.empty();
	};
	const auto success = frame_queue_simulation_cv_.wait_for(lk, timeout, has_item);
	if (success)
	{
		const auto frame = frame_queue_simulation_.front();
		frame_queue_simulation_.pop();
		const auto actual_pattern = frame.pattern;
		const auto action = scope_action::set_bg_for_this_channel == frame_meta_data_before_acquire.action;
		const auto prototype = get_prepared_image(actual_pattern, camera_configuration_.aoi_index, camera_configuration_.bin_index, action);
		{
			const auto sensor_size = this->get_sensor_size(camera_configuration_);
			if (static_cast<const frame_size&>(prototype) != sensor_size )
			{
				qli_runtime_error("Pattern Generation Error");
			}
		}

		//also note that in certain pipelines the data is written in place
		const auto samples = prototype.samples();
		output_buffer.resize(samples);
		std::copy(prototype.img.begin(), prototype.img.end(), output_buffer.begin());
		const auto meta_data = frame_meta_data(frame_meta_data_before_acquire, timestamp());
		const auto output_frame = camera_frame(output_buffer.data(), prototype, meta_data);
		if (!output_frame.is_valid())
		{
			qli_runtime_error("Pattern Generation Problem");
		}
		fill_me(output_frame);
	}
	return success;

}

void virtual_camera_device::fix_camera_internal()
{
	//
}

void virtual_camera_device::apply_settings_internal(const camera_config& config)
{
	state_ = camera_mode_idle_camera_device_state.at(config.mode);
}

void virtual_camera_device::set_exposure_internal(const std::chrono::microseconds&)
{

}

void virtual_camera_device::print_debug(std::ostream&)
{
}

void virtual_camera_device::start_capture_internal()
{
	std::unique_lock<std::mutex> lk(frame_queue_simulation_m_);
	//	frame_queue_simulation_ = std::queue<acquisitionSimulation>(); not needed
}

void virtual_camera_device::stop_capture_internal()
{
	std::unique_lock<std::mutex> lk(frame_queue_simulation_m_);
	frame_queue_simulation_ = std::queue<acquisition_simulation>();
}

std::chrono::microseconds virtual_camera_device::get_min_exposure_internal()
{
	return std::chrono::microseconds(2 * 1000);
}

std::chrono::microseconds virtual_camera_device::get_readout_time_internal()
{
	return ms_to_chrono(10);
}

std::chrono::microseconds virtual_camera_device::get_transfer_time_internal()
{
	return ms_to_chrono(8);
}

virtual_camera_type virtual_camera_device::prompt_for_virtual_camera()
{
	std::cout << "Select a Virtual Camera [default = 0]:" << std::endl;
	std::vector<virtual_camera_type> idx_to_type;
	auto count = 0;
	for (auto& item : virtual_camera_settings::settings)
	{
		std::cout << count++ << ": " << item.second.prefix << std::endl;
		idx_to_type.push_back(item.first);
	}
	const auto max_number = idx_to_type.size();
	size_t input;
	do
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::cout << "Select a camera";

	} while ((!(std::cin >> input)) || std::cin.bad() || input < 0 || input >= max_number);

	return idx_to_type.at(input);
}

bool virtual_camera_device::register_resource()
{
	{
		const auto success = QResource::registerResource(virtual_camera_device::resource_file_name);
		if (!success)
		{
			QMessageBox::information(nullptr, "Resource File", QString("Can't find %1").arg(virtual_camera_device::resource_file_name));
			qli_runtime_error("Can't find resource file");
		}
	}
	return true;
}

virtual_camera_device::virtual_camera_device(virtual_camera_type camera_type, const int camera_idx, QObject* parent) : camera_device(camera_device_features(true, true, true,  true, camera_contrast_features()), camera_idx, parent), pattern_count(0)
{
	if (camera_type == virtual_camera_type::prompt)
	{
		camera_type = prompt_for_virtual_camera();
	}

	exposure_ = ms_to_chrono(10);
	const auto frame_set = make_aois(camera_type);
	common_post_constructor();
	make_virtual_images(frame_set);
}

void virtual_camera_device::set_cooling_internal(const bool enable)
{
	std::cout << (enable ? "Enabling " : "Disabling ") << "cooler on virtual camera " << enable << std::endl;
}