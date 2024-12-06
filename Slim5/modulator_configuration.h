#pragma once
#ifndef MODULATOR_CONFIGURATION_H
#define MODULATOR_CONFIGURATION_H
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <unordered_map>
#include "common_limits.h"
#include "frame_size.h"
#include <array>
#include <chrono>

#include "approx_equals.h"
#include "double_spin_box_settings.h"

enum class slm_mode { single_shot, two_shot_lcvr, slim, qdic, darkfield, custom_patterns, unset };
struct slm_mode_setting
{
	int patterns;
	std::string label;
	typedef std::unordered_map<slm_mode, const slm_mode_setting> slm_mode_setting_map;
	static const slm_mode_setting_map settings;
	slm_mode_setting(const int patterns, const std::string& label) : patterns(patterns), label(label)
	{
	}
};

enum class slm_pattern_mode { donut, file, checkerboard, alignment, count };
typedef std::unordered_map<slm_pattern_mode, const std::string> slm_pattern_mode_names_map;
extern const slm_pattern_mode_names_map slm_pattern_mode_names;


enum class darkfield_mode { dots, dots_with_ring_psi, dots_with_dic_psi, pdots_with_dic_psi, rings, rings_with_psi, rings_with_dic_psi, rings_with_dic_psi2 };
struct darkfield_mode_settings
{
	std::string label;
	bool is_four_frame_psi;
	slm_pattern_mode modulator_mode;
	darkfield_mode_settings(const std::string& label, const bool is_four_frame_psi, const slm_pattern_mode modulator_mode) : label(label), is_four_frame_psi(is_four_frame_psi), modulator_mode(modulator_mode) {}
	typedef std::unordered_map < darkfield_mode, darkfield_mode_settings > darkfield_mode_settings_holder;
	const static darkfield_mode_settings_holder settings;
};

struct illumination_power_settings
{
	float illumination_power, brightfield_scale_factor;
	illumination_power_settings() noexcept :illumination_power_settings(0, 0) {}
	illumination_power_settings(const float illumination_power, const float brightfield_scale_factor) noexcept : illumination_power(illumination_power), brightfield_scale_factor(brightfield_scale_factor) {}
	[[nodiscard]] bool item_approx_equals(const illumination_power_settings& b) const noexcept
	{
		return approx_equals(illumination_power, b.illumination_power) && approx_equals(brightfield_scale_factor, b.brightfield_scale_factor);
	}
	[[nodiscard]] bool operator== (const illumination_power_settings& b) const noexcept
	{
		return  exactly_equals(illumination_power, b.illumination_power) && exactly_equals(brightfield_scale_factor, b.brightfield_scale_factor);
	}
	[[nodiscard]] bool is_complete() const noexcept;
};

struct darkfield_pattern_settings
{
	float width_na;
	float ref_ring_na, objective_na, max_na;
	enum class darkfield_display_align_mode { darkfield, align_ref_ring_na, align_objective_na, align_max_na };
	darkfield_display_align_mode darkfield_display_mode;
	typedef std::unordered_map<darkfield_display_align_mode, const std::string> darkfield_pattern_settings_map;
	const static darkfield_pattern_settings_map darkfield_display_mode_settings;
	bool invert_modulator_x, invert_modulator_y;
	darkfield_pattern_settings(const float width_na, const float ref_ring_na, const float objective_na, const float max_na, const darkfield_display_align_mode darkfield_display_mode, const bool invert_modulator_x, const bool  invert_modulator_y) noexcept :
		width_na(width_na), ref_ring_na(ref_ring_na), objective_na(objective_na), max_na(max_na), darkfield_display_mode(darkfield_display_mode), invert_modulator_x(invert_modulator_x), invert_modulator_y(invert_modulator_y)
	{}
	darkfield_pattern_settings() noexcept : darkfield_pattern_settings(0, 0, 0, 0, darkfield_display_align_mode::darkfield, false, false) {};

	[[nodiscard]] bool is_complete() const noexcept;

	[[nodiscard]] bool item_approx_equals(const darkfield_pattern_settings& b) const noexcept
	{
		return approx_equals(width_na, b.width_na)
			&& approx_equals(ref_ring_na, b.ref_ring_na) && approx_equals(objective_na, b.objective_na) && approx_equals(max_na, b.max_na)
			&& darkfield_display_mode == b.darkfield_display_mode && invert_modulator_x == b.invert_modulator_x && invert_modulator_y == b.invert_modulator_y;
	}
	[[nodiscard]] bool operator== (const darkfield_pattern_settings& b) const noexcept
	{
		return  exactly_equals(width_na, b.width_na)
			&& exactly_equals(ref_ring_na, b.ref_ring_na) && exactly_equals(objective_na, b.objective_na) && exactly_equals(max_na, b.max_na)
			&& darkfield_display_mode == b.darkfield_display_mode && invert_modulator_x == b.invert_modulator_x && invert_modulator_y == b.invert_modulator_y;
	}
	[[nodiscard]] bool operator!= (const darkfield_pattern_settings& b) const noexcept
	{
		return !(*this == b);
	}

};

struct distorted_donut
{
	float x_center, y_center, inner_diameter, outer_diameter, ellipticity_e, ellipticity_f;
	bool pair;
	float x_center_2;
	float y_center_2;
	distorted_donut(const float x_center, const float y_center, const float inner_diameter, const float outer_diameter, const float ellipticity_e, const float ellipticity_f, const bool pair = false, const float x_center_2 = 0.0, const float y_center_2 = 0.0) noexcept : x_center(x_center), y_center(y_center), inner_diameter(inner_diameter), outer_diameter(outer_diameter), ellipticity_e(ellipticity_e), ellipticity_f(ellipticity_f), pair(pair), x_center_2(x_center_2), y_center_2(y_center_2)
	{

	}

	distorted_donut() noexcept : distorted_donut(0, 0, 0, 0, 0, 0) {}
	[[nodiscard]] bool is_complete() const noexcept;

	[[nodiscard]] bool item_approx_equals(const distorted_donut& b) const noexcept
	{
		return
			approx_equals(x_center, b.x_center) &&
			approx_equals(y_center, b.y_center) &&
			approx_equals(inner_diameter, b.inner_diameter) &&
			approx_equals(outer_diameter, b.outer_diameter) &&
			approx_equals(ellipticity_e, b.ellipticity_e) &&
			approx_equals(ellipticity_f, b.ellipticity_f);
	}
	[[nodiscard]] bool operator== (const distorted_donut& b) const noexcept
	{
		return x_center == b.x_center &&
			y_center == b.y_center &&
			inner_diameter == b.inner_diameter &&
			outer_diameter == b.outer_diameter &&
			ellipticity_e == b.ellipticity_e &&
			ellipticity_f == b.ellipticity_f;
	}
	[[nodiscard]] bool operator!= (const distorted_donut& b) const noexcept
	{
		return !(*this == b);
	}
};


struct psi_function_pair
{
	float top, bot, constant;
	psi_function_pair() noexcept : psi_function_pair(0, 0, 0) {}
	psi_function_pair(const float top, const float bot, const float constant) noexcept : top(top), bot(bot), constant(constant) {}
	[[nodiscard]] bool operator== (const psi_function_pair& b) const noexcept
	{
		return b.top == top && b.bot == bot && b.constant == constant;
	}

	[[nodiscard]] bool item_approx_equals(const psi_function_pair& b) const noexcept
	{
		//maybe some std::tie
		return  approx_equals(top, b.top)
			&& approx_equals(bot, b.bot)
			&& approx_equals(constant, b.constant);
	}
	const static double_spin_box_settings spin_box_settings;
	[[nodiscard]] bool is_complete() const noexcept
	{
		const auto blank = psi_function_pair();
		return !(*this == blank);
	}
};

struct slm_levels
{
	float slm_value, slm_background;
	[[nodiscard]] bool operator== (const slm_levels& b) const noexcept
	{
		return b.slm_value == slm_value && b.slm_background == slm_background;
	}

	[[nodiscard]] bool is_valid() const noexcept
	{
		const auto valid_range = [](const float value)
		{
			return value >= 0 && value <= std::numeric_limits<unsigned char>::max();
		};
		return valid_range(slm_value) && valid_range(slm_background);
	}

	[[nodiscard]] bool item_approx_equals(const slm_levels& b) const noexcept
	{
		return approx_equals(slm_value, b.slm_value) && approx_equals(slm_background, b.slm_background);
	}
	slm_levels(const float slm_value, const float slm_background) noexcept : slm_value(slm_value), slm_background(slm_background)
	{

	}
	slm_levels() noexcept : slm_levels(0, 0) {};
};

struct phase_shift_pattern : slm_levels
{
	slm_pattern_mode pattern_mode;
	std::string filepath;
	[[nodiscard]] bool operator== (const phase_shift_pattern& b) const noexcept
	{
		return static_cast<const slm_levels&>(*this) == b && b.filepath == filepath && pattern_mode == b.pattern_mode;
	}

	[[nodiscard]] bool item_approx_equals(const phase_shift_pattern& b) const noexcept
	{
		return static_cast<const slm_levels&>(*this).item_approx_equals(b) && b.filepath == filepath && pattern_mode == b.pattern_mode;
	}
	phase_shift_pattern() noexcept : phase_shift_pattern("", 0, 0, slm_pattern_mode::donut) {};
	explicit phase_shift_pattern(const std::string& filepath, const float slm_value, const float slm_background, const slm_pattern_mode mode) :slm_levels(slm_value, slm_background), pattern_mode(mode), filepath(filepath) {};
	explicit phase_shift_pattern(const slm_levels& levels) : phase_shift_pattern("", levels.slm_value, levels.slm_background, slm_pattern_mode::donut) {}
};

typedef boost::container::small_vector< psi_function_pair, max_samples_per_pixel > psi_function_pairs;

struct per_pattern_modulator_settings : phase_shift_pattern, distorted_donut
{
	psi_function_pairs weights;
	per_pattern_modulator_settings() noexcept : per_pattern_modulator_settings(phase_shift_pattern(), distorted_donut(), psi_function_pairs({ psi_function_pair() })) {}
	per_pattern_modulator_settings(const phase_shift_pattern& phase_shift_pattern, const distorted_donut& dot_info, const psi_function_pairs& weights) noexcept : phase_shift_pattern(phase_shift_pattern), distorted_donut(dot_info), weights(weights) {}

	[[nodiscard]] static bool psi_function_pairs_approx_equals(const psi_function_pairs& a, const psi_function_pairs& b)
	{
		const auto predicate = [](const psi_function_pair& a, const psi_function_pair& b)
		{
			return a.item_approx_equals(b);
		};
		return std::equal(a.begin(), a.end(), b.begin(), b.end(), predicate);
	}
	[[nodiscard]] bool item_approx_equals(const per_pattern_modulator_settings& b) const noexcept
	{
		return phase_shift_pattern::item_approx_equals(b)
			&& distorted_donut::item_approx_equals(b)
			&& psi_function_pairs_approx_equals(this->weights, b.weights);
	}
	[[nodiscard]] bool operator== (const per_pattern_modulator_settings& b) const noexcept
	{
		return weights == b.weights && static_cast<const phase_shift_pattern&>(*this) == b && static_cast<const distorted_donut&>(*this) == b;
	}
	void set_samples_per_pixel(const int samples) noexcept
	{
		weights.resize(samples);
	}
	[[nodiscard]] bool is_valid() const noexcept;
	void assert_valid() const;

};

typedef boost::container::small_vector<per_pattern_modulator_settings, typical_psi_patterns> per_pattern_modulator_settings_patterns;

struct four_frame_psi_setting : slm_levels
{
	psi_function_pairs weights;
	[[nodiscard]] bool is_single_channel() const noexcept
	{
		return weights.size() == 1;
	}
	four_frame_psi_setting(const slm_levels& slm_levels, const psi_function_pairs& weights) :slm_levels(slm_levels), weights(weights)
	{
	}
	four_frame_psi_setting() noexcept :four_frame_psi_setting(slm_levels(), { psi_function_pair() }) {}
	[[nodiscard]] bool is_valid() const noexcept;
	[[nodiscard]] bool item_approx_equals(const four_frame_psi_setting& b) const
	{
		const auto predicate = [](const psi_function_pair& a, const psi_function_pair& b)
		{
			return a.item_approx_equals(b);
		};
		const auto weights_equal = std::equal(weights.begin(), weights.end(), b.weights.begin(), b.weights.end(), predicate);
		return weights_equal && slm_levels::item_approx_equals(b);
	}
};

struct modulator_configuration : distorted_donut, darkfield_pattern_settings, illumination_power_settings
{
	typedef std::array<four_frame_psi_setting, typical_psi_patterns> four_frame_psi_settings;
	four_frame_psi_settings four_frame_psi;

	[[nodiscard]] static bool four_frame_psi_setting_holder_approx_equals(const four_frame_psi_settings& a, const four_frame_psi_settings& b) noexcept
	{
		const auto predicate = [](const four_frame_psi_setting& item_a, const four_frame_psi_setting& item_b)
		{
			return item_a.item_approx_equals(item_b);
		};
		return std::equal(a.begin(), a.end(), b.begin(), b.end(), predicate);
	}
	//Not using QSettings because that would pull in too many libraries...
	std::chrono::microseconds cycle_internal_delay_us;
	float voltage_max;
	[[nodiscard]] bool valid_voltage() const noexcept
	{
		return std::isnormal(voltage_max) && voltage_max > 0;
	}
	constexpr static float default_max_voltage = 5.0f;
	static std::chrono::microseconds cycle_internal_delay_us_default() noexcept
	{
		return std::chrono::milliseconds(0);
	}
	modulator_configuration(const distorted_donut& beam_settings, const darkfield_pattern_settings& darkfield_pattern_settings, const four_frame_psi_settings& four_frame_psi_setting_holder, const illumination_power_settings& illumination_power, const float voltage_max) noexcept : distorted_donut(beam_settings), darkfield_pattern_settings(darkfield_pattern_settings), illumination_power_settings(illumination_power), four_frame_psi(four_frame_psi_setting_holder), cycle_internal_delay_us(cycle_internal_delay_us_default()), voltage_max(voltage_max)
	{
	}

	modulator_configuration() noexcept : modulator_configuration(distorted_donut(), darkfield_pattern_settings(), four_frame_psi_settings(), illumination_power_settings(), 0.0f)
	{
	}

	[[nodiscard]] bool item_approx_equals(const modulator_configuration& b) const noexcept
	{
		//maybe some std::tie
		return
			distorted_donut::item_approx_equals(b)
			&& darkfield_pattern_settings::item_approx_equals(b)
			&& illumination_power_settings::item_approx_equals(b)
			&& four_frame_psi_setting_holder_approx_equals(this->four_frame_psi, b.four_frame_psi)
			&& cycle_internal_delay_us == b.cycle_internal_delay_us
			&& approx_equals(voltage_max, b.voltage_max);
	}
	[[nodiscard]] bool operator== (const modulator_configuration& b) const noexcept
	{
		//maybe some std::tie
		return
			static_cast<const distorted_donut&>(*this) == b
			&& static_cast<const darkfield_pattern_settings&>(*this) == b
			&& static_cast<const illumination_power_settings&>(*this) == b
			&& four_frame_psi == b.four_frame_psi
			&& cycle_internal_delay_us == b.cycle_internal_delay_us
			&& voltage_max == b.voltage_max;
	}
	[[nodiscard]] bool operator!= (const modulator_configuration& b) const noexcept
	{
		return !(*this == b);
	}

	[[nodiscard]] float brightfield_diameter() const noexcept
	{
		return (objective_na / ref_ring_na) * reference_diameter();
	}

	[[nodiscard]] float reference_diameter() const noexcept
	{
		return outer_diameter;
	}

	[[nodiscard]] float darkfield_max_diameter() const  noexcept
	{
		return (max_na / ref_ring_na) * reference_diameter();
	}

	[[nodiscard]] float darkfield_point_width() const  noexcept
	{
		return (width_na / ref_ring_na) * reference_diameter();
	}

	[[nodiscard]] distorted_donut get_alignment_donut(darkfield_display_align_mode mode) const;

	[[nodiscard]] bool is_valid() const noexcept;
};

struct per_modulator_saveable_settings : modulator_configuration
{
	const static int illumination_idx = 1;
	per_pattern_modulator_settings_patterns patterns;
	std::string file_path_basedir;
	bool is_alignment;
	static bool per_pattern_modulator_settings_patterns_approx_equals(const per_pattern_modulator_settings_patterns& a, const per_pattern_modulator_settings_patterns& b) noexcept
	{
		const auto predicate = [](const per_pattern_modulator_settings& a, const per_pattern_modulator_settings& b)
		{
			return a.item_approx_equals(b);
		};
		const auto equal = std::equal(a.begin(), a.end(), b.begin(), b.end(), predicate);
		return equal;
	}

	[[nodiscard]] bool item_approx_equals(const per_modulator_saveable_settings& b) const noexcept
	{

		return modulator_configuration::item_approx_equals(b) && per_pattern_modulator_settings_patterns_approx_equals(this->patterns, b.patterns) && is_alignment == b.is_alignment;
	}
	[[nodiscard]] bool operator== (const per_modulator_saveable_settings& b) const noexcept;
	[[nodiscard]] bool operator!= (const per_modulator_saveable_settings& b) const noexcept
	{
		return !(*this == b);
	}

	[[nodiscard]] bool is_valid() const;
	per_modulator_saveable_settings(const modulator_configuration& modulator_configuration, const per_pattern_modulator_settings_patterns& per_pattern_modulator_settings_patterns, const bool alignment, const std::string& file_path_basedir = "") noexcept : modulator_configuration(modulator_configuration), patterns(per_pattern_modulator_settings_patterns), file_path_basedir(file_path_basedir), is_alignment(alignment) {}
	per_modulator_saveable_settings() noexcept : per_modulator_saveable_settings(modulator_configuration(), per_pattern_modulator_settings_patterns({ per_pattern_modulator_settings() }), false, "")
	{

	}
	static per_modulator_saveable_settings generate_per_modulator_saveable_settings(int patterns, int samples_per_pixel);
};
typedef boost::container::static_vector<per_modulator_saveable_settings, max_slms > fixed_modulator_settings;
enum class phase_retrieval;
typedef boost::container::static_vector<frame_size, max_slms> slm_dimensions;

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(slm_pattern_mode)
Q_DECLARE_METATYPE(slm_mode)
Q_DECLARE_METATYPE(darkfield_mode)
Q_DECLARE_METATYPE(darkfield_pattern_settings::darkfield_display_align_mode)
#endif

#endif