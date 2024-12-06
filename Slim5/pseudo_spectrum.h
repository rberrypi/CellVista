#pragma once
#ifndef PSUEDO_SPECTRUM_GPU_H
#define PSUEDO_SPECTRUM_GPU_H

#include <thrust/device_vector.h>
#include <cuComplex.h>
#include "compute_engine_shared.h"
#include "cufft_shared.h"
class pseudo_spectrum
{
	//its not a real spectrum because we took the log1p of a complex number...
	// All its pronounced 'pseudo'
	thrust::device_vector<cuComplex> img_ft_;
	cufft_wrapper plan;
public:
	void do_pseudo_ft(out_frame in_place, const frame_size& frame);
};

#endif