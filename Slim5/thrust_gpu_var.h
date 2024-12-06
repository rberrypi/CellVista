#pragma once
#ifndef THRUST_GPU_VAR_H
#define THRUST_GPU_VAR_H
#include <thrust/device_vector.h>
class thrust_gpu_var
{
public:
	float get_variance(thrust::device_vector<float>::iterator start, thrust::device_vector<float>::iterator stop) const;
};

#endif