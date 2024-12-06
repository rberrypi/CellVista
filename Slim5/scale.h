#pragma once
#ifndef SCALE_H
#define SCALE_H

template<typename T, typename V >
struct scale
{
	//actually ths whole thing can be replaced by a lookup table!!!!
	//
	//  tell  CUDA that the following code can be executed on the CPU and the GPU
	T from_a, to_a, to_b;
	T factor;
	scale(T from_a, T from_b, T to_a, T to_b) : from_a(from_a), to_a(to_a), to_b(to_b)
	{
		factor = (to_b - to_a) / (from_b - from_a);
	}
	__host__ __device__  V  operator()(const T& convert_me) const
	{
		// this can be optomized, for example division is bad
		auto val = to_a + (convert_me - from_a) * factor;
		return val;
	}
};

#endif
