#include "device_factory.h"
#include "compute_engine_shared.h"
#include "qli_runtime_error.h"
background_update_functors device_factory::get_background_update_functors()
{
	//todo add preflight that backgrounds do, actually match
	//Each functor will update for whatever it can, if the size or scope don't match it won't update
	const dpm_bg_update_functor dpm_functor = [&](const dpm_settings& new_settings, const int channel_idx)
	{
		static_cast<dpm_settings&>(route.ch.at(channel_idx)) = new_settings;
	};
	const slim_update_functor slim_functor = [&](const slim_bg_settings& new_bg, const int channel_idx)
	{
		static_cast<slim_bg_settings&>(route.ch.at(channel_idx)) = new_bg;
	};
	const phase_update_functor phase_functor = [&](const camera_frame_internal& new_background)
	{
		route.ch.at(new_background.channel_route_index).load_background(new_background, true);
	};
	background_update_functors functors = { dpm_functor , slim_functor,  phase_functor };
	return functors;
}