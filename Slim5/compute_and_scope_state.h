#pragma once
#ifndef COMPUTE_AND_SCOPE_STATE_H
#define COMPUTE_AND_SCOPE_STATE_H
#include "render_settings.h"
#include "phase_processing.h"
#include "instrument_configuration.h"
#include "camera_config.h"
#include "capture_cycle.h"
struct camera_frame_internal;
struct dpm_settings;
struct background_frame;
struct band_pass_settings
{
	bool do_band_pass, remove_dc;
	float min_dx, max_dx;
	float min_dy, max_dy;
	void set_min_dr(const float dr) noexcept
	{
		min_dx = dr;
		min_dy = dr;
	}
	void set_max_dr(const float  dr) noexcept
	{
		max_dx = dr;
		max_dy = dr;
	}

	[[nodiscard]] bool is_isotropic() const noexcept
	{
		return min_dy == min_dx && max_dx == max_dy;
	}
	band_pass_settings(const float  min_dx, const float  max_dx, const float  min_dy, const float  max_dy, const bool remove_dc, const bool do_band_pass) noexcept : do_band_pass(do_band_pass), remove_dc(remove_dc), min_dx(min_dx), max_dx(max_dx), min_dy(min_dy), max_dy(max_dy) {}
	band_pass_settings(const float  min_dx, const float  max_dx, const bool remove_dc, const bool do_band_pass) noexcept : do_band_pass(do_band_pass), remove_dc(remove_dc), min_dx(min_dx), max_dx(max_dx), min_dy(min_dx), max_dy(max_dx) {}
	band_pass_settings() noexcept : band_pass_settings(0, 0, 0, 0, false, false) {}
	bool operator== (const band_pass_settings& b) const noexcept
	{
		return do_band_pass == b.do_band_pass && remove_dc == b.remove_dc && min_dx == b.min_dx && max_dx == b.max_dx && min_dy == b.min_dy && max_dy == b.max_dy;
	}

	[[nodiscard]] bool item_approx_equals(const band_pass_settings& b) const noexcept
	{
		return do_band_pass == b.do_band_pass && remove_dc == b.remove_dc && approx_equals(min_dx, b.min_dx) && approx_equals(max_dx, b.max_dx) && approx_equals(min_dy, b.min_dy) && approx_equals(max_dy, b.max_dy);
	}

};
struct slim_bg_settings
{
	float slim_bg_value;// I also don't think a collision can happen with this, although maybe when we retrieve the old value but I think reads are orders on the x86
	[[nodiscard]] bool is_slim_bg_settings_valid() const noexcept
	{
		return std::isfinite(slim_bg_value);
	}
	explicit slim_bg_settings(const float slim_bg_value) noexcept : slim_bg_value(slim_bg_value) {}
	slim_bg_settings() noexcept : slim_bg_settings(std::numeric_limits<float>::quiet_NaN()) {}
	bool operator== (const slim_bg_settings& b) const noexcept
	{
		return slim_bg_value == b.slim_bg_value;
	}
	[[nodiscard]] bool item_approx_equals(const slim_bg_settings& b) const noexcept
	{
		return approx_equals(slim_bg_value, b.slim_bg_value);
	}
};
struct material_info
{
	float n_cell, n_media, mass_inc, obj_height;
	material_info() noexcept : material_info(0, 0, 0, 0.0) {}

	material_info(const float n_media, const float n_cell, const float mass_inc, const float obj_height) noexcept : n_cell(n_cell), n_media(n_media), mass_inc(mass_inc), obj_height(obj_height)
	{
	}
	[[nodiscard]] bool operator== (const material_info& b) const noexcept
	{
		return n_cell == b.n_cell && n_media == b.n_media && mass_inc == b.mass_inc && obj_height == b.obj_height;
	}

	[[nodiscard]] bool item_approx_equals(const material_info& b) const noexcept
	{
		return approx_equals(n_cell, b.n_cell) && approx_equals(n_media, b.n_media) && approx_equals(mass_inc, b.mass_inc) && approx_equals(obj_height, b.obj_height);
	}
	[[nodiscard]] bool operator!= (const material_info& b) const noexcept
	{
		return !(*this == b);
	}
};

struct compute_and_scope_settings : render_settings, microscope_light_path, camera_config, band_pass_settings, slim_bg_settings, processing_quad, material_info
{
	void load_background(const camera_frame_internal& input, bool merge);
	void clear_background();
	typedef  std::shared_ptr<background_frame> background_frame_ptr;
	background_frame_ptr background_;
	//as opposed to per pattern
	float z_offset;
	int current_pattern;
	std::string label_suffix;
	[[nodiscard]] bool item_approx_equals(const compute_and_scope_settings& b) const noexcept
	{
		//check underlying buffer pointer
		return approx_equals(z_offset, b.z_offset) && current_pattern == b.current_pattern && background_ == b.background_ && render_settings::item_approx_equals(b) && microscope_light_path::item_approx_equals(b) && static_cast<const camera_config&>(*this) == b && band_pass_settings::item_approx_equals(b) && slim_bg_settings::item_approx_equals(b) && static_cast<const processing_quad&>(*this) == b && material_info::item_approx_equals(b);
	}
	static std::string fixup_label_suffix(phase_retrieval phase_retrieval, const std::string& label);
	void fixup_label_suffix(const std::string& label);
	//
	compute_and_scope_settings(const processing_quad& live_compute_options, const render_settings& render_settings, const microscope_light_path& microscope_light_path, const camera_config& camera_config, const band_pass_settings& band_pass_settings, const slim_bg_settings& slim_bg_settings, const background_frame_ptr& bg_buffer, const material_info& material_info, const std::string& label_suffix = "") :render_settings(render_settings),
		microscope_light_path(microscope_light_path), camera_config(camera_config), band_pass_settings(band_pass_settings), slim_bg_settings(slim_bg_settings), processing_quad(live_compute_options), material_info(material_info), background_(bg_buffer), z_offset(0), current_pattern(0), label_suffix(label_suffix)
	{
	}
	compute_and_scope_settings() noexcept :compute_and_scope_settings(processing_quad(), render_settings(), microscope_light_path(), camera_config(), band_pass_settings(), slim_bg_settings(), background_frame_ptr(), material_info(), "") {}
	//
	void assert_validity() const;
	void fixup_label_suffix();
	[[nodiscard]] capture_iterator_view iterator(int frames) const;
	[[nodiscard]] bool is_direct_write() const;
	[[nodiscard]] bool is_native_sixteen_bit()  const noexcept;
	[[nodiscard]] bool is_valid() const noexcept;
	[[nodiscard]] std::string get_label_short() const;
	[[nodiscard]] std::string get_label_long() const;
	//
	[[nodiscard]] int modulator_patterns(int frames) const;
	[[nodiscard]] int output_files_per_compute(int frames, bool is_live) const;
	[[nodiscard]] int slm_pattern_for_live_mode(int current_pattern) const noexcept;

};

struct live_compute_options final
{
	enum class background_show_mode { regular=0, show_bg=1, show_bg_subtracted=2, set_bg};
	background_show_mode show_mode;
	bool is_live;
	live_compute_options() noexcept : show_mode(background_show_mode::regular), is_live(false) {}
	live_compute_options(const bool is_live, const background_show_mode show_mode) noexcept : show_mode(show_mode), is_live(is_live) {}
};

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(band_pass_settings)
Q_DECLARE_METATYPE(slim_bg_settings)
Q_DECLARE_METATYPE(material_info)
#endif

#endif