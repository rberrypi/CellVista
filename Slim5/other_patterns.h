#pragma once
#ifndef OTHER_PATTERNS_H
#define OTHER_PATTERNS_H
#include "compute_engine_shared.h"
struct channel_settings;
class other_patterns
{
	const static int preferred_sample = 0;//we're going to get rid of this eventually
	static void non_int_c1(out_frame out, in_frame a, in_frame b, in_frame  c, in_frame d, const channel_settings& in);
	static void non_int_c3(out_frame out, in_frame a, in_frame b, in_frame  c, in_frame d, const channel_settings& in);
	static void ac_dc_c1(out_frame out, in_frame a, in_frame b, in_frame  c, in_frame d, const channel_settings& in);
	static void ac_dc_c3(out_frame out, in_frame a, in_frame b, in_frame  c, in_frame d, const channel_settings& in);
public:
	//todo these will get unified
	static void ac_dc(out_frame out, in_frame a, in_frame b, in_frame  c, in_frame d, const channel_settings& in, int samples_per_pixel);

	static void for_intensity(out_frame out, in_frame a, in_frame b, in_frame  c, in_frame d, const channel_settings& in);
	static void pass_thru(out_frame out, in_frame a);
};

#endif