#include "stdafx.h"
#include "live_settings_store.h"
#include "device_factory.h"
#include "camera_device.h"
#include "slm_device.h"
#include "scope.h"
#include <iostream>
#include <algorithm>
#include "qli_runtime_error.h"

live_gui_settings live_gui_settings::get_default_live_gui_settings(const int scope_channel, const processing_double& settings, const bool is_ft)
{
	const auto& default_camera = D->cameras.front();
	const auto demosaic_mode = camera_chroma_setting::settings.at(default_camera->chroma).preferred_demosaic_mode;
	const auto denoise_off = denoise_mode::off;
	const processing_quad quad(settings.retrieval, settings.processing, demosaic_mode, denoise_off);
	const processing_quad default_quad;
	const auto quad_settings = quad.is_supported_quad() ? quad : default_quad;
	const auto slm_count = D->get_slm_count();
	const auto samples_per_pixel = D->max_samples_per_pixels();
	auto gui_setting = static_cast<live_gui_settings>(channel_settings::generate_test_channel(quad_settings, slm_count, samples_per_pixel));
	gui_setting.scope_channel = scope_channel;
	gui_setting.do_ft = is_ft;
#if _DEBUG
	if (!gui_setting.is_valid())
	{
		qli_runtime_error("Something Terrible");
	}
#endif
	return gui_setting;
}

live_channels live_settings_store::get_default_contrast_settings()
{
	live_channels channels;
	constexpr auto phase_channel = scope_channel_drive_settings::phase_channel_idx, fl_channel = 1, f2_channel = 2, f3_channel = 3, f4_channel = 4;
	//first level
	channels[0] = live_gui_settings::get_default_live_gui_settings(phase_channel, { phase_retrieval::camera, phase_processing::raw_frames }, false);
	channels[1] = live_gui_settings::get_default_live_gui_settings(phase_channel, { phase_retrieval::glim, phase_processing::phase }, false);
	channels[2] = live_gui_settings::get_default_live_gui_settings(phase_channel, { phase_retrieval::slim, phase_processing::phase }, false);
	channels[3] = live_gui_settings::get_default_live_gui_settings(phase_channel, { phase_retrieval::slim_demux, phase_processing::phase }, false);
	//second level
	channels[4] = live_gui_settings::get_default_live_gui_settings(fl_channel, { phase_retrieval::camera, phase_processing::raw_frames }, false);
	channels[5] = live_gui_settings::get_default_live_gui_settings(f2_channel, { phase_retrieval::camera, phase_processing::raw_frames }, false);
	channels[6] = live_gui_settings::get_default_live_gui_settings(f3_channel, { phase_retrieval::camera, phase_processing::raw_frames }, false);
	channels[7] = live_gui_settings::get_default_live_gui_settings(f4_channel, { phase_retrieval::camera, phase_processing::raw_frames }, false);
	channels[8] = live_gui_settings::get_default_live_gui_settings(phase_channel, { phase_retrieval::slim, phase_processing::phase }, true);
	//
#if _DEBUG
	for (auto channel_idx = 0; channel_idx < channels.size(); ++channel_idx)
	{
		const auto is_valid = channels.at(channel_idx).is_valid();
		if (!is_valid)
		{
			const auto busted_channel = "Channel " + std::to_string(channel_idx) + " is busted";
			qli_runtime_error(busted_channel);
		}
	}
#endif
	return channels;
}

const static auto channel_warning = "Warning, can't find: ";

// template<class T>
// constexpr const T& clamp(const T& v, const T& lo, const T& hi)
// {
// 	return (v < lo) ? lo : (hi < v) ? hi : v;
// }

std::string live_settings_store::get_channel_setting_name(const int channel_idx) noexcept
{
	return "channel_" + std::to_string(channel_idx) + ".json";
}


live_settings_store::live_settings_store() noexcept:default_contrast_settings_(get_default_contrast_settings())
{
	//Load
	for (auto i = 0; i < default_contrast_settings_.size(); ++i)
	{
		const auto name = get_channel_setting_name(i);
		bool okay;
		auto channel = live_gui_settings::read(name, okay);
		current_contrast_settings_.at(i) = okay ? channel : default_contrast_settings_.at(i);
	}
	//Apply a few post processing hacks
	const auto condenser_nac_limits = D->scope->chan_drive->get_condenser_na_limit();
	const auto samples = D->max_samples_per_pixels();
	for (auto idx = 0; idx < default_contrast_settings_.size(); ++idx)
	{
		//match limits
		auto& current_setting = current_contrast_settings_.at(idx);
		auto& default_setting = default_contrast_settings_.at(idx);
		//Fixup NAC to match requirements, this is a dirty hack
		current_setting.nac = std::clamp(current_setting.nac, condenser_nac_limits.nac_min, condenser_nac_limits.nac_max);
		default_setting.nac = std::clamp(default_setting.nac, condenser_nac_limits.nac_min, condenser_nac_limits.nac_max);
		//expand channels to match the camera, for example when settings differ
		current_setting.set_samples_per_pixel(samples);
		if (!current_setting.is_valid())
		{
			current_setting = default_setting;
		}
#if _DEBUG
		{
			if (!current_setting.is_valid() || !default_setting.is_valid())
			{
				qli_runtime_error("Oh Nope");
			}
		}
#endif
	}

}

live_settings_store::~live_settings_store()
{
	for (auto i = 0; i < default_contrast_settings_.size(); ++i)
	{
		const auto name = get_channel_setting_name(i);
		try
		{
			const auto success = current_contrast_settings_.at(i).write(name);
			Q_UNUSED(success);
		}
		catch (...)
		{
			std::cout << channel_warning << name << std::endl;
		}
	}
}