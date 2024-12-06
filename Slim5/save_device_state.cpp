#include "stdafx.h"

#include "save_device_state.h"
#include "scope.h"
#include "camera_device.h"
#include "slm_device.h"

save_position_scope::save_position_scope(microscope* scope) : scope(scope), pos_scope(scope->get_state(true))
{
}

save_position_scope::~save_position_scope()
{
	scope->move_to(microscope_move_action(pos_scope, ms_to_chrono(0)), false);
}

save_position_cameras::save_position_cameras(const std::vector<camera_device*>& cameras)
{
	for (auto& camera : cameras)
	{
		camera_state_pair pair;
		pair.camera = camera;
		pair.camera_config = camera->get_camera_config();
		pair.exposure = camera->get_exposure();
		configs.push_back(pair);
	}
}

save_position_cameras::~save_position_cameras()
{
	for (auto& config : configs)
	{
		auto* cam = config.camera;
		cam->apply_settings(config.camera_config);
		cam->set_exposure(config.exposure);
	}
}

save_slm_positions::save_slm_positions(slm_holder* slms) : state(slms->get_modulator_states()), slms(slms)
{
}

save_slm_positions::~save_slm_positions()
{
	const auto initialized = state.front().is_initialized();
	if (initialized)
	{
		slms->load_modulator_states(state);		
	}
}