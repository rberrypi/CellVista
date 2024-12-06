#pragma once
#ifndef SCOPE_COMPUTE_SETTINGS_H
#define SCOPE_COMPUTE_SETTINGS_H
#include "approx_equals.h"
#include "frame_size.h"
#include "common_limits.h"
#include <array>
struct dpm_settings
{
	//specifies the DPM rectangle
	bool dpm_snap_bg;
	int dpm_phase_left_column, dpm_phase_top_row, dpm_phase_width;
	int dpm_amp_left_column, dpm_amp_top_row, dpm_amp_width;
	bool operator== (const dpm_settings& b) const noexcept
	{
		return dpm_snap_bg == b.dpm_snap_bg && dpm_phase_left_column == b.dpm_phase_left_column && dpm_phase_top_row == b.dpm_phase_top_row && dpm_phase_width == b.dpm_phase_width
			&& dpm_amp_left_column == b.dpm_amp_left_column && dpm_amp_top_row == b.dpm_amp_top_row && dpm_amp_width == b.dpm_amp_width;
	}
	bool operator!=(const dpm_settings& b) const noexcept
	{
		return !(*this == b);
	}
	[[nodiscard]] bool is_valid() const noexcept;
	[[nodiscard]] bool dpm_phase_is_valid() const noexcept;//aka something that can be shown in the GUI configuration
	[[nodiscard]] bool dpm_phase_is_complete() const noexcept;
	[[nodiscard]] bool dpm_amp_is_valid() const noexcept;
	[[nodiscard]] bool dpm_amp_is_complete() const noexcept;
	[[nodiscard]] bool fits_in_frame(const frame_size& frame) const;
	dpm_settings(const int dpm_phase_left_column, const int dpm_phase_top_row, const int dpm_phase_width) noexcept: dpm_settings(dpm_phase_left_column, dpm_phase_top_row, dpm_phase_width, dpm_phase_left_column, dpm_phase_top_row, dpm_phase_width) {}
	dpm_settings() noexcept: dpm_settings(0, 0, 0, 0, 0, 0) {}
	dpm_settings(const int dpm_phase_left_column, const int dpm_phase_top_row, const int dpm_phase_width, const int dpm_amp_left_column, const int dpm_amp_top_row, const int dpm_amp_width) noexcept:
		dpm_snap_bg(false), dpm_phase_left_column(dpm_phase_left_column), dpm_phase_top_row(dpm_phase_top_row), dpm_phase_width(dpm_phase_width), dpm_amp_left_column(dpm_amp_left_column), dpm_amp_top_row(dpm_amp_top_row), dpm_amp_width(dpm_amp_width)
	{

	}

	[[nodiscard]] std::string get_dpm_string() const noexcept;
};

struct pixel_dimensions
{
	float pixel_ratio, coherence_length;
	pixel_dimensions(const float coherence_length, const float pixel_ratio) noexcept: pixel_ratio(pixel_ratio), coherence_length(coherence_length) {}
	pixel_dimensions() noexcept:pixel_dimensions(0, 0) {}

	[[nodiscard]] bool item_approx_equals(const pixel_dimensions& b) const noexcept
	{
		return approx_equals(pixel_ratio, b.pixel_ratio) && approx_equals(coherence_length, b.coherence_length);
	}
	[[nodiscard]]  bool operator== (const pixel_dimensions& b) const noexcept
	{
		return pixel_ratio == b.pixel_ratio && coherence_length == b.coherence_length;
	}
	[[nodiscard]] bool operator!= (const pixel_dimensions& b) const noexcept
	{
		return !(*this == b);
	}
	[[nodiscard]] bool is_complete() const noexcept
	{
		return std::isnormal(pixel_ratio) && std::isnormal(coherence_length);
	}
};

struct qdic_scope_settings
{
	float qsb_qdic_shear_angle, qsb_qdic_shear_dx;
	qdic_scope_settings(const float qsb_qdic_shear_angle, const float qsb_qdic_shear_dx) noexcept: qsb_qdic_shear_angle(qsb_qdic_shear_angle), qsb_qdic_shear_dx(qsb_qdic_shear_dx) {}
	qdic_scope_settings() noexcept: qdic_scope_settings(0, 0) {}

	[[nodiscard]] bool item_approx_equals(const qdic_scope_settings& b) const noexcept
	{
		return approx_equals(qsb_qdic_shear_angle, b.qsb_qdic_shear_angle) && approx_equals(qsb_qdic_shear_dx, b.qsb_qdic_shear_dx);
	}
	[[nodiscard]] bool operator== (const qdic_scope_settings& b) const noexcept
	{
		return qsb_qdic_shear_angle == b.qsb_qdic_shear_angle && qsb_qdic_shear_dx == b.qsb_qdic_shear_dx;
	}
	[[nodiscard]] bool operator!= (const qdic_scope_settings& b) const noexcept
	{
		return !(*this == b);
	}
};

typedef std::array<float, max_samples_per_pixel>  wave_length_package;
struct scope_compute_settings : pixel_dimensions, qdic_scope_settings
{
	//awful name
	float objective_attenuation, stage_overlap;
	const static float max_wavelength;
	wave_length_package wave_lengths;
	scope_compute_settings(const float attenuation, const float stage_overlap, const pixel_dimensions& compute_dimensions, const qdic_scope_settings& qdic_settings, const wave_length_package& wavelengths) noexcept:
		pixel_dimensions(compute_dimensions), qdic_scope_settings(qdic_settings), objective_attenuation(attenuation), stage_overlap(stage_overlap), wave_lengths(wavelengths)
	{

	}
	scope_compute_settings() noexcept: scope_compute_settings(0, 0, pixel_dimensions(), qdic_scope_settings(), wave_length_package()) {
	}
	[[nodiscard]] bool operator== (const scope_compute_settings& b) const noexcept
	{
		return objective_attenuation == b.objective_attenuation && stage_overlap == b.stage_overlap && wave_lengths == b.wave_lengths && static_cast<const pixel_dimensions&>(*this) == b && static_cast<const qdic_scope_settings&>(*this) == b;
	}

	[[nodiscard]] bool item_approx_equals(const scope_compute_settings& b) const noexcept
	{
		const auto predicate = [](const float& a, const float& b)
		{
			return approx_equals(a, b);
		};
		return approx_equals(objective_attenuation, b.objective_attenuation) &&
			approx_equals(stage_overlap, b.stage_overlap) &&
			approx_equals(objective_attenuation, b.objective_attenuation) &&
			std::equal(wave_lengths.begin(), wave_lengths.end(), b.wave_lengths.begin(), b.wave_lengths.end(), predicate) && pixel_dimensions::item_approx_equals(b) && qdic_scope_settings::item_approx_equals(b);
	}
	[[nodiscard]] bool operator!= (const scope_compute_settings& b) const noexcept
	{
		return !(*this == b);
	}

	[[nodiscard]] bool is_complete() const noexcept;
};


#endif