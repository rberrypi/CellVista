#pragma once
#ifndef SLM_STATE_H
#define SLM_STATE_H
#include "modulator_configuration.h"

enum class slm_trigger_mode { software, hardware };
struct slm_state : per_modulator_saveable_settings
{
	slm_trigger_mode internal_mode = slm_trigger_mode::software;
	const static int uninitialized_position = -1;
	int frame_number = uninitialized_position;
	int slm_port = uninitialized_position;
	[[nodiscard]] bool is_initialized() const noexcept
	{
		return frame_number != slm_state::uninitialized_position;
	}
};
typedef boost::container::small_vector<slm_state, typical_psi_patterns > slm_states;
#endif