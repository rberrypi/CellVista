#include "stdafx.h"
#include "compute_and_scope_state.h"
#include <sstream>
#include <iostream>
#include "device_factory.h"
#include "qli_runtime_error.h"
#include "scope.h"

bool compute_and_scope_settings::is_native_sixteen_bit() const noexcept
{
	const auto raw_frames = is_raw_frame();
	const auto skip_bandpass = !do_band_pass;
	const auto no_denoise = denoise == denoise_mode::off;
	const auto not_spectrum = !do_ft;
	return raw_frames && skip_bandpass && no_denoise && not_spectrum;
}

bool compute_and_scope_settings::is_direct_write() const
{
	//this means we can directly write the pointer rather than do computation. 
	const auto is_native_sixteen = is_native_sixteen_bit();
	const auto skip_demosaic = demosaic == demosaic_mode::no_processing;
	const auto no_background = !background_;
	return is_native_sixteen && skip_demosaic && no_background;
}

[[nodiscard]] int compute_and_scope_settings::modulator_patterns(const int frames) const
{
	const auto patterns = phase_retrieval_setting::settings.at(retrieval).modulator_patterns();
	const auto final_patterns = patterns == pattern_count_from_file ? frames : patterns;
	return final_patterns;
}

capture_iterator_view compute_and_scope_settings::iterator(const int frames) const
{
	const auto pattern_limit = modulator_patterns(frames);
	const auto denoise_limit = denoise_setting::settings.at(denoise).patterns;
	const auto  limits = cycle_position(denoise_limit, pattern_limit);
	return{ limits,retrieval,processing,denoise };
}

int compute_and_scope_settings::output_files_per_compute(const int frames, const bool is_live) const
{
	const auto patterns = modulator_patterns(frames);
	const auto per_demosaic = frames_per_demosaic(is_live);
	const auto compute_patterns = 1;//AKA 1 pattern per modulation
	const auto files = processing != phase_processing::raw_frames ? compute_patterns : patterns * per_demosaic;
	if (files < 1)
	{
		qli_runtime_error("Something Wong");
	}
	return files;
}

[[nodiscard]] int compute_and_scope_settings::slm_pattern_for_live_mode(const int current_pattern) const noexcept
{
	return modulates_slm() && !is_raw_frame()  ? current_pattern : this->current_pattern;
}

std::string compute_and_scope_settings::get_label_long() const
{
	std::stringstream ss;
	if (do_ft)
	{
		ss << "Spectrum ";
	}
	ss << get_label_short();
	if (processing != phase_processing::raw_frames)
	{
		const auto& label = phase_processing_setting::settings.at(processing).label;
		ss << " " << label;
	}
	if (denoise != denoise_mode::off)
	{
		const auto suffix = denoise_setting::settings.at(denoise).label;
		ss << " " << suffix;
	}
	return  ss.str();
}

std::string compute_and_scope_settings::get_label_short() const
{
	const auto get_modifiers = [&]
	{
		std::string modifiers;
		if (processing == phase_processing::raw_frames && retrieval != phase_retrieval::camera)
		{
			modifiers += " Raw";
		}
		if (do_ft)
		{
			modifiers += " FT";
		}
		return modifiers;
	};
	const auto get_prefix = [&]
	{
		const auto is_raw_frames = processing == phase_processing::raw_frames;
		const auto is_camera = retrieval == phase_retrieval::camera;
		if (is_camera && is_raw_frames)
		{
			//no modulation, raw frames, aka DAPI/FITC
			return D->scope->chan_drive->channel_names.at(scope_channel);
		}
		if (!is_camera && is_raw_frames)
		{
			//modulation but raw frames, call it slim, etc
			return phase_retrieval_setting::settings.at(retrieval).label;
		}
		if (!is_camera && !is_raw_frames)
		{
			// ie slim with phase, call it SLIM when phase, something else when other thing is set
			return processing == phase_processing::phase ? phase_retrieval_setting::settings.at(retrieval).label : phase_processing_setting::settings.at(processing).label;
		}
		if (is_camera && !is_raw_frames)
		{
			// no modulation but something other than raw frames, this should never happen
			// processing = mutual?, retrieval = 0
			std::cout << "Processing " << static_cast<int>(processing) << "," << static_cast<int>(retrieval) << std::endl;
			qli_runtime_error("Invalid Config?");
		}
		qli_runtime_error("All control paths return a value");
	};
	auto return_me = get_prefix() + get_modifiers();
	return return_me;
}

bool scope_compute_settings::is_complete() const noexcept
{
	return objective_attenuation > 0 && stage_overlap > 0 && pixel_dimensions::is_complete();
}

bool compute_and_scope_settings::is_valid() const noexcept
{
	const auto label_suffix_filled_out = !this->label_suffix.empty();
	const auto supported_mode = is_supported_quad();
	return label_suffix_filled_out && supported_mode;
}

std::string compute_and_scope_settings::fixup_label_suffix(const phase_retrieval phase_retrieval, const std::string& label)
{
	const auto is_camera = phase_retrieval == phase_retrieval::camera;
	if (is_camera)
	{
		return label;
	}
	const auto& settings = phase_retrieval_setting::settings.at(phase_retrieval);
	return settings.label;
}

void compute_and_scope_settings::fixup_label_suffix(const std::string& label)
{
	label_suffix = fixup_label_suffix(retrieval, label);
}

void compute_and_scope_settings::assert_validity() const
{
#if _DEBUG
	if (!is_valid())
	{
		qli_runtime_error("Invalid Compute Settings");
	}
#endif
}

void compute_and_scope_settings::fixup_label_suffix()
{
	const auto& channel_names = D->scope->get_channel_settings_names();
	const auto& channel_name = channel_names.at(scope_channel);
	this->label_suffix = compute_and_scope_settings::fixup_label_suffix(retrieval, channel_name.toStdString());
}

