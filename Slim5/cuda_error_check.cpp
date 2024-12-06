#include "stdafx.h"
#include "cufft_shared.h"
#include "cuda_error_check.h"
#include <cuda_runtime_api.h>
#include <sstream>
#include <cufft.h>
#include "cufft_error_check.h"
#include "qli_runtime_error.h"

cuda_memory_info get_cuda_memory_info()
{
	cuda_memory_info info = { 0,0 };
	CUDASAFECALL(cudaDeviceSynchronize());
	CUDASAFECALL(cudaMemGetInfo(&info.free_byte, &info.total_byte));
	return info;
}
void cuda_memory_debug(const char* file, const int line)
{
	CUDASAFECALL(cudaDeviceSynchronize());
	const auto info = get_cuda_memory_info();
	const auto free_db = static_cast<double>(info.free_byte);
	const auto total_db = static_cast<double>(info.total_byte);
	const auto used_db = total_db - free_db;
	const auto percentage = used_db / total_db;
	const auto used_ram_mb = used_db / 1024.0 / 1024.0;
	const auto free_ram_mb = free_db / 1024.0 / 1024.0;
	const auto total_ram_mb = total_db / 1024.0 / 1024.0;
	printf("%s:%d -> GPU mem usage %0.2f: %.0f / %.0f\n", file, line, percentage, used_ram_mb, total_ram_mb);
	fflush(stdout);
	CUDASAFECALL(cudaDeviceSynchronize());
}

void cuda_debug_sync(const char* file, const int line)
{
#if _DEBUG
	cuda_safe_call(cudaDeviceSynchronize(), file, line);
	cuda_safe_call(cudaDeviceSynchronize(), file, line);
#endif
}

void cuda_safe_call(const cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		std::stringstream error_msg;
		error_msg << "cuda_safe_call() failed at " << file << ":" << line << ":" << cudaGetErrorString(err);
		qli_runtime_error(error_msg.str());
	}
}

static const char* cufft_get_error_enum(const cufftResult_t error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";
	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";
	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";
	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";
	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";
	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";
	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";
	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";
	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	case CUFFT_INCOMPLETE_PARAMETER_LIST: break;
	case CUFFT_INVALID_DEVICE: break;
	case CUFFT_PARSE_ERROR: break;
	case CUFFT_NO_WORKSPACE: break;
	case CUFFT_NOT_IMPLEMENTED: break;
	case CUFFT_LICENSE_ERROR: break;
	default: break;
	}
	return "<unknown>";
}

void cufft_safe_call(const cufftResult_t err, const char* file, const int line)
{
	if (CUFFT_SUCCESS != err)
	{
		std::stringstream error_msg;
		error_msg << "CufftSafeCall() failed at " << file << ":" << line << ":" << cufft_get_error_enum(err);
		qli_runtime_error(error_msg.str());
	}
}
