#pragma once
#ifndef BACKGROUND_UPDATE_FUNCTORS_H
#define BACKGROUND_UPDATE_FUNCTORS_H
#include "compute_and_scope_state.h"

struct camera_frame_internal;
struct dpm_settings;
struct slim_bg_settings;

typedef std::function<void(const dpm_settings&, int)> dpm_bg_update_functor;
typedef std::function<void(const slim_bg_settings&, int)> slim_update_functor;
typedef std::function<void(const camera_frame_internal&)> phase_update_functor;
struct background_update_functors final
{
	dpm_bg_update_functor dpm_functor;
	slim_update_functor slim_functor;
	phase_update_functor phase_functor;
};

#endif