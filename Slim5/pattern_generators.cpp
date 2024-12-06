#include "stdafx.h"

#include <iostream>

#include "settings_file.h"
#include "qli_runtime_error.h"

struct illumination_info
{
	float x, y, intensity;
	bool is_brightfield;
	illumination_info(const float x, const float y, const float intensity, const bool is_brightfield) noexcept :x(x), y(y), intensity(intensity), is_brightfield(is_brightfield) {}
	illumination_info() noexcept : illumination_info(0, 0, 0, false) {}
};


void settings_file::fill_four_frame_psi(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, const slm_pattern_mode slm_mode, const bool is_illumination, const int patterns, int samples_per_pixel)
{
	changed.resize(patterns);
	for (auto i = 0; i < patterns; ++i)
	{
		auto& pattern = changed.at(i);
		auto setting = per_modulator_saveable_settings.four_frame_psi.at(i);
		setting.weights.resize(samples_per_pixel);
		const auto donut = distorted_donut(per_modulator_saveable_settings);
		const auto slm_illumination = per_modulator_saveable_settings.brightfield_scale_factor * per_modulator_saveable_settings.illumination_power;
		constexpr auto slm_illumination_off = 0.0f;
		const phase_shift_pattern illumination_levels({ slm_illumination,slm_illumination_off });
		const auto slm_foreground = setting.slm_value;
		const auto slm_background = slm_mode == slm_pattern_mode::donut ? per_modulator_saveable_settings.four_frame_psi.front().slm_value : setting.slm_background;
		const phase_shift_pattern psi_levels({ slm_foreground,slm_background });
		auto shifts = is_illumination ? illumination_levels : psi_levels;
		shifts.pattern_mode = is_illumination ? slm_pattern_mode::checkerboard : slm_mode;
		pattern = per_pattern_modulator_settings(shifts, donut, setting.weights);
		pattern.assert_valid();
	}
}


void settings_file::fill_dot_list(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, const int samples, const bool is_illumination, const darkfield_mode darkfield_mode, const int samples_per_pixel)
{
	//Step 1: generate points
	thread_local std::vector<illumination_info> points;
	const auto e = per_modulator_saveable_settings.ellipticity_e;
	const auto f = per_modulator_saveable_settings.ellipticity_f;
	const auto brightness = per_modulator_saveable_settings.illumination_power;

	const auto bright_field_max = per_modulator_saveable_settings.brightfield_diameter();
	const auto dark_field_max = per_modulator_saveable_settings.darkfield_max_diameter();
	const auto x1 = per_modulator_saveable_settings.x_center;
	const auto y1 = per_modulator_saveable_settings.y_center;
	{
		points.resize(0);

		const auto dot_diameter = 2 * samples + 1;
		const auto beam_step = dark_field_max / static_cast<float>(dot_diameter - 1);
		const auto x_factor = per_modulator_saveable_settings.invert_modulator_x && !is_illumination ? (-1) : 1;
		const auto y_factor = per_modulator_saveable_settings.invert_modulator_y && !is_illumination ? (-1) : 1;
		for (auto y = -samples; y <= samples; ++y)
		{
			for (auto x = -samples; x <= samples; ++x)
			{
				const auto x_pos = x_factor * x * beam_step;
				const auto y_pos = y_factor * y * beam_step;
				//while it might be outside the SLM, we still need them to match the other modulator, so lets leave them in
				//auto inside_slm = ((x_pos < slm_size.width && x_pos >= 0) && (y_pos < slm_size.height && y_pos >= 0));
				//if (inside_slm)
				{
					const auto distance = hypot(x_pos * e, y_pos * f);
					const auto is_inside = samples <= hypot(samples,samples);
					const auto is_brightfield = distance <= (bright_field_max / 2);
					if (is_inside)
					{
						const auto x_pos_final = x_pos + x1;
						const auto y_pos_final = y_pos + y1;
						const auto brightness_level = is_brightfield ? brightness * per_modulator_saveable_settings.brightfield_scale_factor : brightness;
						points.emplace_back(x_pos_final, y_pos_final, brightness_level, is_brightfield);
					}
				}
			}
		}
	};
	std::cout << "Dot List Length " << points.size() << std::endl;
	//
	const auto dot_radius = per_modulator_saveable_settings.darkfield_point_width();
	const auto& darkfield_settings = darkfield_mode_settings::settings.at(darkfield_mode);
	changed.resize(0);
	//
	const auto illuminations = darkfield_settings.is_four_frame_psi ? 4 : 1;
	for (const auto& point : points)
	{
		//insert same illuminations
		for (auto i = 0; i < illuminations; ++i)
		{
			auto pattern_weights = per_modulator_saveable_settings.four_frame_psi.at(i).weights;
			pattern_weights.resize(samples_per_pixel);
			if (is_illumination)
			{
				const  auto black_background = 0.0f;
				phase_shift_pattern pattern({ point.intensity, black_background });
				const distorted_donut dot(point.x, point.y, 0, dot_radius, e, f);
				const auto dot_setting = per_pattern_modulator_settings(pattern, dot, pattern_weights);
				changed.push_back(dot_setting);
			}
			else
			{
				const auto& slim_ring = static_cast<distorted_donut>(per_modulator_saveable_settings);
				const auto& darkfield_levels = per_modulator_saveable_settings.four_frame_psi;
				const slm_levels levels(darkfield_levels.at(i).slm_value, darkfield_levels.front().slm_value);
				phase_shift_pattern pattern(levels);
				auto modulating_pattern = per_pattern_modulator_settings(pattern, slim_ring, pattern_weights);
				modulating_pattern.pattern_mode = darkfield_settings.modulator_mode;
				changed.push_back(modulating_pattern);
			}
		}
	}
}

void settings_file::fill_paired_dot_list(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, const int samples, const bool is_illumination, const darkfield_mode darkfield_mode, const int samples_per_pixel)
{
	//Step 1: generate points
	thread_local std::vector<std::vector<illumination_info>> points;
	const auto e = per_modulator_saveable_settings.ellipticity_e;
	const auto f = per_modulator_saveable_settings.ellipticity_f;
	const auto brightness = per_modulator_saveable_settings.illumination_power;

	const auto bright_field_max = per_modulator_saveable_settings.brightfield_diameter();
	const auto dark_field_max = per_modulator_saveable_settings.darkfield_max_diameter();
	const auto x1 = per_modulator_saveable_settings.x_center;
	const auto y1 = per_modulator_saveable_settings.y_center;
	{
		points.resize(0);

		const auto dot_diameter = 2 * samples + 1;
		const auto beam_step = dark_field_max / static_cast<float>(dot_diameter - 1);
		const auto x_factor = per_modulator_saveable_settings.invert_modulator_x && !is_illumination ? (-1) : 1;
		const auto y_factor = per_modulator_saveable_settings.invert_modulator_y && !is_illumination ? (-1) : 1;
		for (auto y = -samples; y <= samples; ++y)
		{
			for (auto x = -samples; x <= samples; ++x)
			{
				const auto x_pos = x_factor * x * beam_step;
				const auto y_pos = y_factor * y * beam_step;
				//while it might be outside the SLM, we still need them to match the other modulator, so lets leave them in
				//auto inside_slm = ((x_pos < slm_size.width && x_pos >= 0) && (y_pos < slm_size.height && y_pos >= 0));
				//if (inside_slm)
				{
					const auto distance = hypot(x_pos * e, y_pos * f);
					const auto is_inside = samples <= hypot(samples,samples);
					const auto is_brightfield = distance <= (bright_field_max / 2);
					if (is_inside)
					{
						const auto x_pos_final = x_pos + x1;
						const auto y_pos_final = y_pos + y1;
						const auto brightness_level = is_brightfield ? brightness * per_modulator_saveable_settings.brightfield_scale_factor : brightness;
						std::vector<illumination_info> paired_illumination {
							illumination_info(x_pos_final, y_pos_final, brightness_level, is_brightfield),
							illumination_info(x_pos_final - 2 * x_pos, y_pos_final - 2 * y_pos, brightness_level, is_brightfield),
						};
						points.emplace_back(paired_illumination);
					}
				}
			}
		}
	};
	std::cout << "Paired Dot List Length " << points.size() << std::endl;
	const auto dot_radius = per_modulator_saveable_settings.darkfield_point_width();
	const auto& darkfield_settings = darkfield_mode_settings::settings.at(darkfield_mode);
	changed.resize(0);

	const auto illuminations = darkfield_settings.is_four_frame_psi ? 4 : 1;
	for (const auto& paired_point : points)
	{
		//insert same illuminations
		for (auto i = 0; i < illuminations; ++i)
		{
			auto pattern_weights = per_modulator_saveable_settings.four_frame_psi.at(i).weights;
			pattern_weights.resize(samples_per_pixel);
			if (is_illumination)
			{
				const  auto black_background = 0.0f;
				// new type, paired dots
				auto point = paired_point.at(0);
				auto pair_point = paired_point.at(1);
				phase_shift_pattern pattern({ point.intensity, black_background });
				const distorted_donut pair_dots(point.x, point.y, 0, dot_radius, e, f, true, pair_point.x, pair_point.y);
				// std::cout << pair_dots.x_center << " " << pair_dots.x_center_2 << " " << pair_dots.pair << std::endl;
				const auto pair_dots_setting = per_pattern_modulator_settings(pattern, pair_dots, pattern_weights);
				changed.push_back(pair_dots_setting);
			}
			else
			{
				const auto& slim_ring = static_cast<distorted_donut>(per_modulator_saveable_settings);
				const auto& darkfield_levels = per_modulator_saveable_settings.four_frame_psi;
				const slm_levels levels(darkfield_levels.at(i).slm_value, darkfield_levels.front().slm_value);
				phase_shift_pattern pattern(levels);
				auto modulating_pattern = per_pattern_modulator_settings(pattern, slim_ring, pattern_weights);
				modulating_pattern.pattern_mode = darkfield_settings.modulator_mode;
				changed.push_back(modulating_pattern);
			}
		}
	}
}



void settings_file::fill_circle_list(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, const int samples, bool is_illumination, const darkfield_mode darkfield_mode, int samples_per_pixel)
{
	changed.resize(0);
	//So, we create one "brightfield ring, this is regular SLIM", the rest are increasingly larger darkfield images
	const auto& darkfield_settings = darkfield_mode_settings::settings.at(darkfield_mode);
	const auto width = per_modulator_saveable_settings.darkfield_point_width();
	const auto add_sequence_at_diameter = [&](const float diameter, const bool is_brightfield)
	{
		const auto inner_diameter = diameter - width;
		const auto outer_diameter = diameter + width;
		const auto ellipticity_e = per_modulator_saveable_settings.ellipticity_e;
		const auto ellipticity_f = per_modulator_saveable_settings.ellipticity_f;
		const auto donut = distorted_donut(per_modulator_saveable_settings.x_center, per_modulator_saveable_settings.y_center, inner_diameter, outer_diameter, ellipticity_e, ellipticity_f);
		//hack hack hack
		const auto patterns = darkfield_settings.is_four_frame_psi ? 4 : 1;
		for (auto pattern = 0; pattern < patterns; ++pattern)
		{
			auto psi_settings = per_modulator_saveable_settings.four_frame_psi.at(pattern);
			psi_settings.weights.resize(samples_per_pixel);
			if (is_illumination)
			{
				const auto illumination_level = is_brightfield ? per_modulator_saveable_settings.illumination_power * per_modulator_saveable_settings.brightfield_scale_factor : per_modulator_saveable_settings.illumination_power;
				constexpr auto black_background = 0.0f;
				phase_shift_pattern phase_shift_pattern({ illumination_level,black_background });
				phase_shift_pattern.pattern_mode = slm_pattern_mode::donut;//will always be rings
				per_pattern_modulator_settings setting(phase_shift_pattern, donut, psi_settings.weights);
				changed.push_back(setting);
			}
			else
			{
				phase_shift_pattern phase_shift_pattern(psi_settings);
				phase_shift_pattern.pattern_mode = darkfield_settings.modulator_mode;
				per_pattern_modulator_settings setting(phase_shift_pattern, donut, psi_settings.weights);
				changed.push_back(setting);
			}
		}
	};
	const auto slim_diameter = per_modulator_saveable_settings.reference_diameter();
	add_sequence_at_diameter(slim_diameter, true);
	const auto start_diameter = per_modulator_saveable_settings.brightfield_diameter();
	const auto end_diameter = per_modulator_saveable_settings.darkfield_max_diameter();
	const auto diameter_spacing = (end_diameter - start_diameter) / samples;
	for (auto count = 0; count < samples; ++count)
	{
		const auto diameter = start_diameter + count * diameter_spacing;
		add_sequence_at_diameter(diameter, false);
	}
	//fixup for the GUI
	for (auto& pat : changed)
	{
		pat.inner_diameter = std::max(pat.inner_diameter, 0.0f);
		pat.outer_diameter = std::max(pat.outer_diameter, 0.0f);
	}
}

bool four_frame_psi_setting::is_valid() const noexcept
{
	const auto valid = !weights.empty();
#if _DEBUG
	if (!valid)
	{
		const auto volatile what = 0;
	}
#endif
	return valid;
}

distorted_donut modulator_configuration::get_alignment_donut(const darkfield_display_align_mode mode) const
{

	const auto radius = [&]
	{
		switch (mode)
		{
		case darkfield_display_align_mode::align_ref_ring_na:
		{
			return reference_diameter();
		}
		case darkfield_display_align_mode::align_objective_na:
		{
			return brightfield_diameter();
		}
		case darkfield_display_align_mode::align_max_na:
		{
			return darkfield_max_diameter();
		}
		case darkfield_display_align_mode::darkfield:
		default:
			qli_not_implemented();
		}
	}();
	const auto width = this->darkfield_point_width();
	const auto inner = std::max(0.f, radius - width);
	const auto outside = std::max(0.f, radius + width);

	const distorted_donut donut(x_center, y_center, inner, outside, ellipticity_e, ellipticity_f);
	return donut;
}

void settings_file::regenerate_pattern(const slm_dimensions& slm_dimensions, const int samples_per_pixel)
{
	const auto slm_count = slm_dimensions.size();
	modulator_settings.resize(slm_count);
	for (auto slm_id = 0; slm_id < modulator_settings.size(); ++slm_id)
	{
		auto& modulation = modulator_settings.at(slm_id);
		const auto is_illumination = slm_id == per_modulator_saveable_settings::illumination_idx;
		const auto slm_size = slm_dimensions.at(slm_id);
		const auto& patterns = slm_mode_setting::settings.at(modulator_mode).patterns;
		for (auto& frame : modulation.four_frame_psi)
		{
			frame.weights.resize(samples_per_pixel);
		}
		switch (modulator_mode)
		{
		case slm_mode::slim:
		{
			settings_file::fill_four_frame_psi(modulation.patterns, modulation, slm_size, slm_pattern_mode::donut, is_illumination, patterns, samples_per_pixel);
			break;
		}
		case slm_mode::single_shot:
		case slm_mode::two_shot_lcvr:
		case slm_mode::qdic:
		{
			settings_file::fill_four_frame_psi(modulation.patterns, modulation, slm_size, slm_pattern_mode::checkerboard, is_illumination, patterns, samples_per_pixel);
			break;
		}
		case slm_mode::darkfield:
		{
			if (modulator_mode == slm_mode::darkfield)
			{
				switch (darkfield)
				{
				case darkfield_mode::dots:
					settings_file::fill_dot_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				case darkfield_mode::dots_with_ring_psi:
					settings_file::fill_dot_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				case darkfield_mode::dots_with_dic_psi:
					settings_file::fill_dot_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				case darkfield_mode::pdots_with_dic_psi:
					settings_file::fill_paired_dot_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				case darkfield_mode::rings:
					settings_file::fill_circle_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				case darkfield_mode::rings_with_psi:
					settings_file::fill_circle_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				case darkfield_mode::rings_with_dic_psi:
					settings_file::fill_circle_list(modulation.patterns, modulation, slm_size, darkfield_samples, is_illumination, darkfield, samples_per_pixel);
					break;
				default:
					qli_not_implemented();
				}
				break;
			}
		}
		default:
		{
			qli_not_implemented();
		}

		}
		//draw alignment pattern
		const auto is_darkfield_alignment = modulation.darkfield_display_mode != darkfield_pattern_settings::darkfield_display_align_mode::darkfield;
		if (this->modulator_mode == slm_mode::darkfield && is_darkfield_alignment)
		{
			const auto donut = modulation.get_alignment_donut(modulation.darkfield_display_mode);
			for (auto& pattern : modulation.patterns)
			{
				static_cast<distorted_donut&>(pattern) = donut;
				pattern.pattern_mode = slm_pattern_mode::donut;
				pattern.slm_background = 0;
				pattern.slm_value = 255.f;
			}
		}
		if (modulation.is_alignment)
		{
			for (auto& pattern : modulation.patterns)
			{
				pattern.pattern_mode = slm_pattern_mode::alignment;
			}
		}
#if _DEBUG
		if (!modulation.is_valid())
		{
			qli_runtime_error();
		}
#endif
	}
#if _DEBUG
	if (!is_valid())
	{
		qli_runtime_error("Maybe shouldn't happen");
	}
#endif
}