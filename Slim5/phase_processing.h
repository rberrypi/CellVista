#pragma once
#ifndef PHASE_PROCESSING_H
#define PHASE_PROCESSING_H
#include <string>
#include <unordered_map>
#include <boost/container/static_vector.hpp>
#include <set>
#include "display_settings.h"
#include "slm_state.h"


enum class demosaic_mode { no_processing, rggb_14_native, polarization_0_45_90_135, polarization_0_90, polarization_45_135, count };
enum class phase_retrieval { camera, darkfield, slim, slim_demux, fpm, diffraction_phase, glim, glim_demux, polarizer_demux_single, polarizer_demux_psi, polarizer_demux_two_frame_dpm, custom_patterns, last_retrieval };
enum class live_cycle_mode { every_frame, half_cycle, full_cycle };
enum class phase_processing { phase, diffraction_phase, diffraction_phase_larger, mass, height, non_interferometric, refractive_index, mutual_intensity, raw_frames, degree_of_linear_polarization, angles_of_linear_polarization, stoke_0, stoke_1, stoke_2, quad_pass_through, quad_phase, pol_psi_octo_compute, count };
enum class denoise_mode { off, average, median, hybrid, count };

struct demosaic_setting final
{
	typedef std::set<phase_retrieval> retrieval_modes;
	std::string label;
	bool will_be_color;
	int raw_frames_per_pattern;
	int binning;
	retrieval_modes supported_retrieval_modes;
	typedef std::unordered_map<demosaic_mode, const demosaic_setting > demosaic_settings_map;
	static const demosaic_settings_map info;
	demosaic_setting(const std::string& label, const bool will_be_color, const int raw_frames_per_pattern, const int binning, const retrieval_modes& processing_modes) :
		label(label), will_be_color(will_be_color), raw_frames_per_pattern(raw_frames_per_pattern), binning(binning), supported_retrieval_modes(processing_modes)
	{

	}
};

struct demosaic_requirements
{
	bool requires_a_bayer_for_processing;
	demosaic_mode required_demosaic;
	demosaic_requirements(const bool requires_a_bayer_for_processing, const demosaic_mode required_demosaic) noexcept : requires_a_bayer_for_processing(requires_a_bayer_for_processing), required_demosaic(required_demosaic)
	{

	}

	explicit demosaic_requirements(const demosaic_mode required_demosaic) noexcept : demosaic_requirements(true, required_demosaic) {}
	demosaic_requirements() noexcept : demosaic_requirements(false, demosaic_mode::no_processing) {}
};

struct phase_retrieval_setting_processing : demosaic_requirements
{
	typedef std::set<phase_processing> phase_processing_modes;
	phase_processing_modes supported_processing_modes;
	typedef std::set<denoise_mode> denoise_modes;
	denoise_modes supported_denoise_modes;
	phase_retrieval_setting_processing(const demosaic_requirements& demosaic_requirements, const phase_processing_modes& supported_processing_modes, const denoise_modes& supported_denoise_modes) noexcept :demosaic_requirements(demosaic_requirements), supported_processing_modes(supported_processing_modes), supported_denoise_modes(supported_denoise_modes) {}
};

struct modulator_requirements
{
	slm_mode slm_mode;
	[[nodiscard]] int modulator_patterns() const noexcept
	{
		return slm_mode_setting::settings.at(slm_mode).patterns;
	}
	int processing_patterns;
	live_cycle_mode live_cycle_mode;
	// if you do it on a polarization mode you get out 4 images, if you do it on a regular mode you get out regular images
	bool special_polarizer_paths;
};

struct phase_retrieval_setting final : phase_retrieval_setting_processing, modulator_requirements
{
	std::string label;
	bool has_color_paths;
	[[nodiscard]] phase_processing default_processing_mode() const noexcept
	{
		return *supported_processing_modes.begin();
	}
	typedef std::unordered_map<phase_retrieval, const phase_retrieval_setting> phase_retrieval_settings;
	const static phase_retrieval_settings settings;

	[[nodiscard]] bool do_live(const int current_pattern) const noexcept
	{
		if (live_cycle_mode == live_cycle_mode::every_frame)
		{
			return true;
		}
		if (live_cycle_mode == live_cycle_mode::half_cycle)
		{
			const auto last_pattern = (processing_patterns / 2);
			return (current_pattern % last_pattern) == (last_pattern - 1);
		}
		return (current_pattern % processing_patterns) == (processing_patterns - 1);
	}
	phase_retrieval_setting(const std::string& label, const phase_retrieval_setting_processing& live_compute_options, const modulator_requirements& requirements, const bool has_color_paths) : phase_retrieval_setting_processing(live_compute_options), modulator_requirements(requirements), label(label), has_color_paths(has_color_paths)
	{}
};

struct phase_processing_setting final
{
	[[nodiscard]] display_settings get_display_settings() const noexcept
	{
		return display_settings(display_lut, display_range);
	}
	int display_lut;
	display_range display_range;
	std::string label;
	bool is_a_dpm;
	bool skip_when_testing;
	int upscale;
	bool is_raw_mode;
	static std::unordered_map<phase_processing, phase_processing_setting> settings;
};

struct denoise_setting final
{
	std::string label;
	int patterns;
	typedef std::unordered_map<denoise_mode, denoise_setting > denoise_settings_map;
	const static denoise_settings_map settings;
	static int max_denoise_patterns();
	static int max_denoise_setting_characters();
	typedef boost::container::static_vector<denoise_mode, static_cast<int>(denoise_mode::count)> supported_denoise_modes;
	static const supported_denoise_modes& get_supported_denoise_modes(const phase_retrieval& retrieval)
	{
		if (retrieval == phase_retrieval::custom_patterns)
		{
			const static denoise_setting::supported_denoise_modes options(1, denoise_mode::off);
			return options;
		}
		const static denoise_setting::supported_denoise_modes modes = { denoise_mode::off,denoise_mode::average,denoise_mode::median,denoise_mode::hybrid };
		return modes;
	}
};

struct processing_double
{
	phase_retrieval retrieval;
	phase_processing processing;
	processing_double(const phase_retrieval retrieval, const phase_processing processing) noexcept : retrieval(retrieval), processing(processing) {};
	processing_double() noexcept : processing_double(phase_retrieval::camera, phase_processing::raw_frames) {};
	[[nodiscard]] bool is_raw_frame() const noexcept;
};

struct processing_quad : processing_double
{
	demosaic_mode demosaic;
	denoise_mode denoise;
	processing_quad() noexcept : processing_quad(phase_retrieval::camera, phase_processing::raw_frames, demosaic_mode::no_processing, denoise_mode::off) {};

	processing_quad(const phase_retrieval retrieval, const phase_processing processing, const demosaic_mode demosaic, const denoise_mode denoise_mode) noexcept :
		processing_double{ retrieval, processing }, demosaic(demosaic), denoise(denoise_mode)
	{};
	[[nodiscard]] bool is_supported_quad() const noexcept;
	bool operator== (const processing_quad& b) const noexcept
	{
		return retrieval == b.retrieval && processing == b.processing && demosaic == b.demosaic && denoise == b.denoise;
	}

	[[nodiscard]] int frames_per_demosaic(bool is_live) const;
	[[nodiscard]] bool modulates_slm() const noexcept;
	[[nodiscard]] static bool has_dpm() noexcept;
	[[nodiscard]] static int max_retrieval_input_frames() noexcept;
};

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(denoise_mode)
Q_DECLARE_METATYPE(phase_retrieval)
Q_DECLARE_METATYPE(phase_processing)
Q_DECLARE_METATYPE(demosaic_mode)
#endif
#endif