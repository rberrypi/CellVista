#include "camera_frame.h"
#include "compute_engine_shared.h"
#include "thrust_resize.h"
#include "write_debug_gpu.h"
#include "compute_and_scope_state.h"

struct bg_merger
{
	__host__ __device__ float operator()(const float new_value, const float old) const
	{
		return (old + new_value) / 2;
	}
};

template<typename T>
__global__ void _merge_in(T* inplace, const T* src, const int n)
{
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		inplace[idx] = (inplace[idx] + src[idx]) / 2;
	}
}

void compute_and_scope_settings::clear_background()
{
	background_.reset();
}

void delete_background(background_frame* background)
{
	delete background;
}

void compute_and_scope_settings::load_background(const camera_frame_internal& input, bool merge)
{
	const auto first_set = !background_;
	if (first_set)
	{
		background_ = std::shared_ptr<background_frame>(new background_frame, delete_background);
	}
	const auto& buffer = *input.buffer;
	const auto number_of_elements = input.samples();
	thrust_safe_resize(background_->buffer, number_of_elements);
	if (merge && !first_set)
	{
		static int gridSize, blockSize;
		static auto old_size = (-1);
		const auto size_changed = (number_of_elements != old_size);
		if (size_changed)
		{
			int minGridSize;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, _merge_in<float>, 0, 0));//todo bug here on the type!!!
			gridSize = (number_of_elements + blockSize - 1) / blockSize;
			old_size = number_of_elements;
		}
		auto src = thrust::raw_pointer_cast(buffer.data());		
		auto dst = thrust::raw_pointer_cast(background_->buffer.data());
		_merge_in << < gridSize, blockSize >> > (dst, src, number_of_elements);
	}
	else
	{
		thrust::copy(buffer.begin(), buffer.end(), background_->buffer.begin());
	}
	static_cast<internal_frame_meta_data&>(*background_.get()) = input;
}