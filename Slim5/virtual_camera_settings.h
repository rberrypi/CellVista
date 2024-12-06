#pragma once
#ifndef VIRTUAL_CAMERA_SETTINGS_H
#define VIRTUAL_CAMERA_SETTINGS_H
#include "camera_device.h"
#include "channel_settings.h"
#include "virtual_camera_shared.h"

struct camera_test_vector
{
	//add forced color
	demosaic_mode demosaic;
	phase_retrieval retrieval;
	bool is_forced_color;
	bool operator<(const camera_test_vector& b) const noexcept;

	camera_test_vector() noexcept : camera_test_vector(demosaic_mode::no_processing, phase_retrieval::camera, false) {}

	camera_test_vector(const demosaic_mode demosaic, const phase_retrieval retrieval, const bool is_forced_color) noexcept : demosaic(demosaic), retrieval(retrieval), is_forced_color(is_forced_color)
	{

	}
	//
	typedef std::map<camera_test_vector, virtual_camera_type> camera_test_vectors;
	static camera_test_vectors tests;
};

struct virtual_camera_settings : camera_contrast_features, channel_settings
{
	enum class dpm { yes, no };
	dpm dpm_state;
	enum class camera_kind { four_patterns, calibration };
	camera_kind kind;
	[[nodiscard]] bool pattern_count() const noexcept
	{
		return kind == camera_kind::four_patterns ? typical_psi_patterns : typical_calibration_patterns;
	}
	[[nodiscard]] bool is_dpm() const noexcept
	{
		return dpm_state == dpm::yes;
	}
	[[nodiscard]] bool demo_images_support_resize() const noexcept;
	std::string prefix;
	[[nodiscard]] void bool_verify_resource_path() const;
	virtual_camera_settings(const std::string& prefix, const dpm dpm_state, const camera_contrast_features& camera_contrast_features, const channel_settings& channel_settings, const camera_kind kind = camera_kind::four_patterns) noexcept :
		camera_contrast_features(camera_contrast_features), channel_settings(channel_settings), dpm_state(dpm_state), kind(kind), prefix(prefix)
	{
	}
	virtual_camera_settings() noexcept : virtual_camera_settings("", dpm::no, camera_contrast_features(), channel_settings(), camera_kind::four_patterns) {}
	typedef std::unordered_map<virtual_camera_type, virtual_camera_settings> virtual_camera_settings_map;
	static virtual_camera_settings_map settings;
	static void transfer_settings_to_test_cameras(int slms);
	[[nodiscard]] camera_test_vector get_camera_test_vector() const;
};

#endif