#include "compute_engine.h"
#include "thrust_resize.h"
#include "cuda_error_check.h"
#include <numeric>
#include "channel_settings.h"
template<typename it, typename it_out, typename binary_predicate>
void move_if(it& input, it_out& destination, binary_predicate pred)
{
	auto i = std::stable_partition(input.begin(), input.end(), pred);
	std::move(i, input.end(), std::back_inserter(destination));
	input.erase(i, input.end());
}

template<typename container, typename container_out>
//void move_from_start(container& input, typename std::iterator_traits<container>::difference_type n, container_out& destination)
void move_from_start(container& input, int n, container_out& destination)
{
	auto it = std::next(input.begin(), n);
	std::move(input.begin(), it, std::back_inserter(destination));
	input.erase(input.begin(), it);
}

template<typename T>
__global__ void _denosie_and_scale(T* dst, const T* src, const float scale, const int n)
{
	// Get our global thread ID
	// Make sure we do not go out of bounds
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		dst[idx] = dst[idx] + (src[idx] * scale);
	}
}

void denosie_and_scale(thrust::device_vector<float>& dst_array, thrust::device_vector<float>& src_array, float scale)
{
	//does this work in place?
	static int gridSize, blockSize;
	static auto old_size = (-1);
	const auto input_size = dst_array.size();
	const auto size_changed = (input_size != old_size);
	if (size_changed)
	{
		int minGridSize;//unused?
		CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, _denosie_and_scale<float>, 0, 0));//todo bug here on the type!!!
		gridSize = (input_size + blockSize - 1) / blockSize;
		old_size = input_size;
	}
	auto dst = thrust::raw_pointer_cast(dst_array.data());
	const auto src = thrust::raw_pointer_cast(src_array.data());
	_denosie_and_scale << < gridSize, blockSize >> > (dst, src, scale, input_size);
}

compute_engine::denoise_output_package compute_engine::get_denoised_data(const camera_frame_internal& phase, const channel_settings& settings, bool is_live)
{
	const auto empty_filter_queue = [&] {
		if (!frame_filtering_bank.empty())
		{
			std::unique_lock<std::mutex> lk(output_free_queue_m);
			while (!frame_filtering_bank.empty())
			{
				auto ptr = frame_filtering_bank.front();
				frame_filtering_bank.pop_front();
				output_free_queue.push_back(ptr.buffer);
			}
			output_free_queue_cv.notify_one();
		}
	};
	if (!phase.is_valid())
	{
		return{};
	}
	if (settings.denoise == denoise_mode::off)
	{
		empty_filter_queue();
		return box(phase);
	}
	//
	if (!frame_filtering_bank.empty() && (static_cast<frame_size>(frame_filtering_bank.front()) != phase))
	{
		empty_filter_queue();
	}
	master_alias_check(box(phase));
	frame_filtering_bank.push_back(std::move(phase));
	const auto expected_denoise_pattern_count = denoise_setting::settings.at(settings.denoise).patterns;
	//This can't be right
	const auto expected_compute_items = settings.output_files_per_compute(is_live);//often 1
	const auto denoise_cycle_size = expected_denoise_pattern_count * settings.output_files_per_compute(is_live);
	const auto needs_more_frames = frame_filtering_bank.size() < denoise_cycle_size;
	if (needs_more_frames)
	{
		return{};
	}
	denoise_output_package package;
	const auto expected_patterns = [&] {
		if (settings.processing == phase_processing::raw_frames)
		{
			boost::container::small_vector<int, typical_psi_patterns> expected_patterns(expected_compute_items);
			std::iota(expected_patterns.begin(), expected_patterns.end(), 0);
			return expected_patterns;
		}
		else
		{
			const auto terminal_pattern = phase_retrieval_setting::settings.at(settings.retrieval).processing_patterns - 1;
			boost::container::small_vector<int, typical_psi_patterns> expected_patterns(1, terminal_pattern);
			return expected_patterns;
		}
	};
	//Merge 0,1,2,3,4,5,6
	for (auto pattern_id : expected_patterns())
	{
		auto flush_buffer = [&](std::deque<camera_frame_internal>& to_flush)
		{
			while (!to_flush.empty())
			{
				const auto frame_to_return = to_flush.front();
				to_flush.pop_front();
				output_free_queue.push_back(frame_to_return.buffer);
			}
		};
		std::deque<camera_frame_internal> items_to_merge;
		if (is_live)
		{
			move_from_start(frame_filtering_bank, denoise_cycle_size, items_to_merge);
		}
		else
		{
			const auto move_same_type = [&](const camera_frame_internal& item) {return (item.pattern_idx) != pattern_id; };//
			move_if(frame_filtering_bank, items_to_merge, move_same_type);
		}
		if (items_to_merge.size() < expected_denoise_pattern_count)
		{
			//throw them all back, this happens in the live mode when getting denoise indexes out of order
			std::unique_lock<std::mutex> lk(output_free_queue_m);
			flush_buffer(items_to_merge);
			flush_buffer(frame_filtering_bank);
			return{};
		}
#if _DEBUG
		{
			auto expected_size = items_to_merge[0].n();
			auto better_be_true = std::all_of(items_to_merge.begin(), items_to_merge.end(), [&](const camera_frame_internal& in) {return in.n() == expected_size; });
			if (!better_be_true)
			{
				qli_runtime_error("Logic Error");
			}
		}
#endif
		switch (settings.denoise)
		{
		case denoise_mode::median:
		case denoise_mode::average:
			get_five_tap_filter(*filter_buffer_ptr, items_to_merge, settings.denoise);
			break;
		case denoise_mode::hybrid:
			get_hybrid_filter(*filter_buffer_ptr, items_to_merge);
			break;
		default:
			qli_runtime_error("Logic Error");
		}
		//
		const auto& front = items_to_merge.front();
		const internal_frame_meta_data internal_meta_data(front, front);
		const camera_frame_internal return_me(filter_buffer_ptr, internal_meta_data);//Maybe also merge compute results?
		filter_buffer_ptr = items_to_merge.front().buffer;//note the pointer gets `nuked`, this item will get used in subsequent calculations, this is thread safe because of the mutex :-)
		items_to_merge.pop_front();
		if (is_live)// throw them all back!!! (note we actually already consumed one)
		{
			move_from_start(items_to_merge, items_to_merge.size(), frame_filtering_bank);
		}
		else
		{
			std::unique_lock<std::mutex> lk(output_free_queue_m);
			flush_buffer(items_to_merge);
		}
		//
		package.push_back(return_me);
		if (!items_to_merge.empty())
		{
			qli_runtime_error("Logic Error");
		}
	}
	//
#if _DEBUG
	{
		const auto expected_images = settings.output_files_per_compute(is_live);
		const auto actual_output = package.size();
		if (actual_output != expected_images)
		{
			qli_runtime_error("Logic Error");
		}
	}
#endif
	//
	return package;
}