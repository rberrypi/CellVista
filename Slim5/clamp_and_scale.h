#pragma once
#ifndef CLAMP_AND_SCALE_H
#define CLAMP_AND_SCALE_H

template<typename T, typename V >
struct clamp_n_scale
{
	//actually ths whole thing can be replaced by a lookup table!!!!
	//
	//  tell  CUDA that the following code can be executed on the CPU and the GPU
	T from_a, to_a, to_b;
	T factor;
	clamp_n_scale(T from_a, T from_b, T to_a, T to_b) : from_a(from_a), to_a(to_a), to_b(to_b)
	{
		factor = (to_b - to_a) / (from_b - from_a);
	}
	__host__ __device__  V  operator()(const T& convert_me) const
	{
		// this can be optomized, for example division is bad
		auto val = to_a + (convert_me - from_a) * factor;
		val = (val > to_b) ? to_b : val;//clamp rules
		val = (val < to_a) ? to_a : val;//clamp rules
		return val;
	}
};

#endif
