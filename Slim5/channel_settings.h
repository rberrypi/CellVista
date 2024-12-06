#pragma once
#ifndef CHANNEL_SETTINGS_H
#define CHANNEL_SETTINGS_H
#include "fixed_hardware_settings.h"
#include "compute_and_scope_state.h"
#include "live_gui_settings.h"

struct settings_file;
struct channel_settings : fixed_hardware_settings, live_gui_settings
{
	//
	[[nodiscard]] bool is_valid() const noexcept;
	[[nodiscard]] int output_files_per_compute(bool is_live) const;
	[[nodiscard]] static channel_settings generate_test_channel(const processing_quad& testing_quad);
	channel_settings(const fixed_hardware_settings& fixed_hardware_settings, const live_gui_settings& live_gui_settings) noexcept;
	channel_settings() noexcept: channel_settings(fixed_hardware_settings(), live_gui_settings()) {}
	void fixup_label_suffix();
	void fixup_channel();
	void assert_validity() const;
	//
	[[nodiscard]] capture_iterator_view iterator() const;//alternative to to template out the live_settings<sequence> for example
	void write(const std::string& filename);
	typedef boost::container::small_vector<float, typical_psi_patterns> compensations;
	[[nodiscard]] compensations get_compensations() const;
	[[nodiscard]] size_t bytes_per_capture_item_on_disk() const;
	[[nodiscard]] image_info image_info_per_capture_item_on_disk() const;
	[[nodiscard]] bool has_valid_background() const;
	[[nodiscard]] bool difference_clears_background(const channel_settings& channel_settings) const noexcept;
	[[nodiscard]] bool difference_requires_camera_reload(const channel_settings& channel_settings) const noexcept;
	[[nodiscard]] static channel_settings generate_test_channel(const processing_quad& testing_quad, int slms, int samples_per_pixel);
};
#endif