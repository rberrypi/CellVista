#pragma once
#ifndef POLARIZATION_FILTER_H
#define POLARIZATION_FILTER_H

#include "compute_engine_shared.h"

struct polarization_filters
{
	frame_size merge_quads(out_frame out, in_frame a, in_frame b, in_frame c, in_frame d, const frame_size& size);
	void polarization_merger(out_frame out, in_frame a, in_frame b, in_frame c, in_frame d, phase_processing processing);
};
#endif
