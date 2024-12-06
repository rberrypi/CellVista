#pragma once
#ifndef SLIM_GPU_STRUCTS_H
#define SLIM_GPU_STRUCTS_H
#include <thrust/device_vector.h>
#include "background_update_functors.h"
#include "compute_engine_shared.h"
struct channel_settings;
class slim_gpu_structs
{
	thrust::device_vector<float> extra_;
protected:
	slim_update_functor slim_update;
public:
	void compute_slim_phase(out_frame out, in_frame a, in_frame b, in_frame c, in_frame d, const channel_settings& settings, bool update_bg, int channel_idx, bool is_fpm);
};

#endif