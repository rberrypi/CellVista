#pragma once
#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H
#include <driver_types.h>

struct cuda_memory_info
{
	size_t free_byte, total_byte;
};
//workaorund for a bug
extern cuda_memory_info get_cuda_memory_info();
#define CUDASAFECALL( err ) cuda_safe_call(err, __FILE__, __LINE__ )
void cuda_safe_call(cudaError err, const char* file, int line);

#define CUDA_DEBUG_SYNC(  ) cuda_debug_sync( __FILE__, __LINE__ )
void cuda_debug_sync(const char* file, int line);
void cuda_memory_debug(const char* file, int line);

#endif