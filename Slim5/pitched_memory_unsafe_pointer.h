#pragma once
#ifndef QLI_PITCHED_MEMORY_PTR
#define QLI_PITCHED_MEMORY_PTR
#include "cuda_error_check.h"
#include <boost/core/noncopyable.hpp>
#include <cuda_runtime.h>
#include "frame_size.h"
template<typename T>
struct pitched_memory_unsafe_pointer final : cudaResourceDesc, private boost::noncopyable
{
	pitched_memory_unsafe_pointer() : pitched_memory_unsafe_pointer(frame_size()) {}
	explicit pitched_memory_unsafe_pointer(const frame_size& frame_size) { res.pitch2D.devPtr = nullptr; allocate(frame_size); }
	bool allocate(const frame_size& frame_size)
	{
		const auto frame_size_changed = !(frame_size.width == res.pitch2D.width && frame_size.height == res.pitch2D.height);
		if (frame_size_changed)
		{
			if (res.pitch2D.devPtr != nullptr)
			{
				deallocate();
			}
			std::memset(this, 0, sizeof(pitched_memory_unsafe_pointer));
			CUDASAFECALL(cudaMallocPitch(&res.pitch2D.devPtr, &res.pitch2D.pitchInBytes, frame_size.width * sizeof(T), frame_size.height));
			resType = cudaResourceTypePitch2D;
			res.pitch2D.desc = cudaCreateChannelDesc<T>();
			res.pitch2D.width = frame_size.width;
			res.pitch2D.height = frame_size.height;
		}
		return frame_size_changed;
	}
	void deallocate()
	{
		if (res.pitch2D.devPtr != nullptr)
		{
			CUDASAFECALL(cudaFree(res.pitch2D.devPtr));
			std::memset(this, 0, sizeof(pitched_memory_unsafe_pointer));//makes it a zombie object
		}
	}
	~pitched_memory_unsafe_pointer()
	{
		deallocate();
	}
};
#endif