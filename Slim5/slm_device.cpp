#include "stdafx.h"
#include "slm_device.h"
#include <QImage>
#include <iostream>

#include "qli_runtime_error.h"

slm_device::slm_device(const  int width, const  int height, const bool is_retarder) :
	frame_size{ width,height }, is_retarder(is_retarder)
{
	internal_mode = slm_trigger_mode::software;
}

void slm_device::load_modulator_state(const slm_state& slm_state)
{
	load_patterns(slm_state.slm_port, slm_state);
	set_frame(slm_state.frame_number);
}

namespace geometry
{

	template <class T>
	void fill_circle(T* to_fill, const frame_size& slm_dim, const per_pattern_modulator_settings& el)
	{
		const auto e = el.ellipticity_e;
		const auto f = el.ellipticity_f;
		const auto ir = el.inner_diameter;
		const auto or = el.outer_diameter;
		const auto x1 = el.x_center;
		const auto y1 = el.y_center;
		const auto wid = slm_dim.width;
		const auto hei = slm_dim.height;
		const auto inner = el.slm_value;
		const auto outer = el.slm_background;
		//Only operate only on relevant portion, this might speed up the computation ?
		const auto bar_ir = pow(ir / 2, 2);
		const auto bar_or = pow(or / 2, 2);
		//auto startH = std::max((y1 - Or)*f, 0.f);
		//auto stopH = std::min((y1 + Or)*f, hei*1.f);
		//todo speed up by drawing only the region inside the circle?
		for (auto j = 0; j < hei; j++)
		{
			for (auto i = 0; i < wid; i++)
			{
				//should be a hypotf?
				auto z = pow(e * (i - x1), 2) + pow(f * (j - y1), 2);
				auto outside = z < bar_ir || z > bar_or;
				// incorporating the case of paired-dots (for darkfield only)
				if (el.pair)
				{
					const auto x2 = el.x_center_2;
					const auto y2 = el.y_center_2;
					z = pow(e * (i - x2), 2) + pow(f * (j - y2), 2);
					outside = outside && (z < bar_ir || z > bar_or);
				}
				to_fill[j * wid + i] = outside ? outer : inner;
			}
		}
	}


	template <class T>
	void fill_check_board(T* to_fill, const frame_size& frame, T value_one, T value_two)
	{
		auto smaller = std::min(value_one, value_two);
		auto bigger = std::max(value_one, value_two);
		const auto diff = bigger - smaller;
		const auto width = frame.width;
		const auto height = frame.height;
		if (smaller == bigger)
		{
			for (auto i = 0; i < height * width; i++)
			{
				to_fill[i] = smaller;
			}
		}
		else
		{
			for (auto r = 0; r < height; r++)
			{
				for (auto c = 0; c < width; c++)
				{
					auto idx = c + r * width;
					to_fill[idx] = smaller + diff * (r + c & 1);
				}
			}
		}
	}
}

bool per_modulator_saveable_settings::operator== (const per_modulator_saveable_settings& b) const noexcept
{
	const auto boost_container_hack = [&]
	{
		const auto predicate = [](const per_pattern_modulator_settings& a, const per_pattern_modulator_settings& b) {return a == b; };
		return std::equal(patterns.begin(), patterns.end(), b.patterns.begin(), b.patterns.end(), predicate);
	};
	return static_cast<const modulator_configuration&>(*this) == b && boost_container_hack() && file_path_basedir == b.file_path_basedir;
}

void slm_device::load_pattern(const int pattern_idx)
{
	auto& frame_buffer = frame_data_.at(pattern_idx);
	if (frame_buffer.dirty)
	{
		std::cout << "Rebaking " << pattern_idx << std::endl;
		const auto slm_elements = n();
		frame_buffer.data.resize(slm_elements);
		const auto& pattern = patterns.at(pattern_idx);
		// ReSharper disable once CppLocalVariableMayBeConst
		auto* ptr_to_data = frame_buffer.data.data();
		switch (pattern.pattern_mode)
		{
		case slm_pattern_mode::file:
		{
			const auto full_file = std::filesystem::path(file_path_basedir) / std::filesystem::path(pattern.filepath);
			fill_custom_pattern(ptr_to_data, full_file, static_cast<frame_size>(*this));
			break;
		}
		case slm_pattern_mode::donut:
		{
			geometry::fill_circle(ptr_to_data, *this, pattern);//to slm
			break;
		}
		case slm_pattern_mode::checkerboard:
		{
			const auto pat_f = pattern.slm_value;
			const auto top = static_cast<unsigned char>(std::min(roundf(pat_f), 255.0f));
			const auto bot = static_cast<unsigned char>(std::max(floorf(pat_f), 0.0f));
			geometry::fill_check_board(ptr_to_data, *this, top, bot);
			break;
		}
		case slm_pattern_mode::alignment:
		{
			const auto inner = pattern.slm_value;
			const auto outer = pattern.slm_background;
			const auto inner_c = static_cast<unsigned char>(round(inner));
			const auto outer_c = static_cast<unsigned char>(round(outer));
			fill_symbol(ptr_to_data, outer_c, inner_c, *this);
			break;
		}
		default:
		{
			qli_not_implemented();
		}
		}
		load_frame_internal(pattern_idx);
		//
		frame_buffer.dirty = false;
	}
}

void slm_device::load_patterns(const int slm_index, const per_modulator_saveable_settings& settings)//and FYI dump everything
{
#if _DEBUG
	if (!settings.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	{
		std::lock_guard<std::recursive_mutex> lk(protect_patterns_);
		const auto sequence_size = settings.patterns.size();
		frame_data_.resize(sequence_size);
		for (auto& meta_data : frame_data_)
		{
			meta_data.dirty = true;
		}
		slm_port = slm_index;
		static_cast<per_modulator_saveable_settings&>(*this) = settings;
		frame_number = uninitialized_position;//<- in an unknown state, right now because the frame hasn't been "set"
	}
#if _DEBUG
	{
		const auto setting_size = settings.patterns.size();
		const auto frame_size = frame_data_.size();
		if (setting_size != frame_size)
		{
			auto current_patterns = static_cast<per_modulator_saveable_settings&>(*this);
			qli_runtime_error("Loading Error");
		}
	}
#endif
}

int slm_device::get_frame_number()
{
	//probably needs a mutex, actually not because its atomic
	std::lock_guard<std::recursive_mutex> lk(protect_patterns_);
	return frame_number;
}

int slm_device::get_frame_number_total()
{
	std::lock_guard<std::recursive_mutex> lk(protect_patterns_);
	return static_cast<int>(frame_data_.size());
}

void slm_device::set_frame_await(const int frame_number, const std::chrono::microseconds& slm_delay_ms, const bool wait_on)
{
	//wait_on = false -> same as async
	//wait_on = true - > enforce timeout in slm_delay_ms
	static auto last_call = ms_to_chrono(0);
	if (wait_on)
	{
		const auto post_delay = get_frame_number() != frame_number ? slm_delay_ms : ms_to_chrono(0);
		const auto elapsed = timestamp() - last_call;
		if (elapsed < slm_delay_ms)
		{
			const auto remaining = slm_delay_ms - elapsed;
			windows_sleep(remaining);
		}
		set_frame(frame_number);
		windows_sleep(post_delay);
	}
	else
	{
		set_frame(frame_number);
	}
	last_call = timestamp();
}


bool slm_device::set_frame(const int frame_number)
{
	//prevents it from being called twice...
	std::lock_guard<std::recursive_mutex> lk(protect_patterns_);
	//notice the settings changed makes this become uninitialized 
	if (frame_number == this->frame_number || frame_number == uninitialized_position)
	{
		//don't crash, but fyi you can't set this
		return true;
	}
	if (frame_number >= frame_data_.size())
	{
		//we can't really put an error here due to the live mode
		return false;
	}
	load_pattern(frame_number);
	set_frame_internal(frame_number);
	this->frame_number = frame_number;
	return true;
}

const unsigned char* slm_device::get_frame(const int frame)
{
	std::lock_guard<std::recursive_mutex> lk(protect_patterns_);
	const auto frames = static_cast<int>(frame_data_.size());
	load_pattern(frame);//get what should actually be in there ;-)
	return abs(frame) < frames ? frame_data_.at(frame).data.data() : nullptr;
}

void slm_device::toggle_mode(const slm_trigger_mode mode)
{
	if (!has_high_speed_mode() && slm_trigger_mode::hardware == mode)
	{
		qli_runtime_error("");
	}
	toggle_mode_internal(mode);
	internal_mode = mode;
}

	void slm_device::fix_hardware_trigger()
	{
		qli_not_implemented();
	}

void slm_device::hardware_trigger_sequence(const size_t capture_items, const channel_settings& channel_settings)
{
	if (internal_mode != slm_trigger_mode::hardware)
	{
		qli_runtime_error("");
	}
	hardware_trigger_sequence_internal(capture_items, channel_settings);
}

void slm_device::hardware_trigger_sequence_internal(const size_t, const channel_settings&)
{
	qli_runtime_error("");
}

void slm_device::toggle_mode_internal(slm_trigger_mode)
{
}

slm_state slm_device::get_modulator_state() const
{
	return static_cast<slm_state>(*this);
}