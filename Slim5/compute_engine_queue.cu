#include "compute_engine.h"
#include "thrust_resize.h"
#include <numeric>
#include "cuda_error_check.h"
#include "write_debug_gpu.h"
#include "time_slice.h"
#include "ml_shared.h"
#include "channel_settings.h"
//#include "write_debug_gpu.h"
#include <boost/container/static_vector.hpp>
//#include <boost/range/algorithm.hpp>
// ReSharper disable once CppMemberFunctionMayBeStatic
void compute_engine::master_alias_check(const denoise_output_package& frames) const
{
	Q_UNUSED(frames);
#ifdef _DEBUG
	{
		//Nothing should ever alias, ever
		//SO, this is kinda messed up because it needs to grab mutex to work, but that would dead lock, we can only use it in debug mode (aka fuck it race conditions, and please don't distribute it) you cucks
		{
			std::vector<thrust::device_vector<float>*> ptrs;
			for (auto input : frames)
			{
				ptrs.push_back(input.buffer);
			}
			for (auto input : output_free_queue)
			{
				ptrs.push_back(input);
			}
			for (auto input : output_destination_queue)
			{
				ptrs.push_back(input.buffer);
			}
			for (auto input : frame_filtering_bank)
			{
				ptrs.push_back(input.buffer);
			}
			ptrs.push_back(filter_buffer_ptr);
			ptrs.push_back(shifter_buffer_ptr);
			ptrs.push_back(decomplexify_buffer_ptr);
			//
			std::sort(ptrs.begin(), ptrs.end());
			auto pos = std::adjacent_find(ptrs.begin(), ptrs.end());
			if (pos != ptrs.end())
			{
				qli_runtime_error("No Aliasing");
			}
		}
	}
#endif
}

void compute_engine::flush_and_reset()
{
	std::unique_lock<std::mutex> output_free_queue_m_lk(output_free_queue_m);
	std::unique_lock<std::mutex> output_destination_queue_m_lk(output_destination_queue_m);
	while (!output_destination_queue.empty())
	{
		const auto top = output_destination_queue.front();
		output_free_queue.push_back(top.buffer);
		output_destination_queue.pop_front();
	}
	while (!frame_filtering_bank.empty())
	{
		const auto top = frame_filtering_bank.front();
		output_free_queue.push_back(top.buffer);
		frame_filtering_bank.pop_front();
	}
}

void compute_engine::kill_queues()
{
	if (!disconnect_queues_)
	{
		disconnect_queues_ = true;
		//notify queques here
		output_free_queue_cv.notify_one();
		output_destination_queue_cv.notify_one();
	}
}

compute_engine::~compute_engine()
{
	//possible memory leak?
	kill_queues();
}

camera_frame_internal_buffer* compute_engine::get_out_frame()
{
	std::unique_lock<std::mutex> lk(output_free_queue_m);
	output_free_queue_cv.wait(lk, [&] {return (!output_free_queue.empty()) || disconnect_queues_; });
	if (disconnect_queues_)
	{
		return nullptr;
	}
	const auto output_frame = output_free_queue.front();
	output_free_queue.pop_front();
	return output_frame;
}

camera_frame_internal compute_engine::get_phase(const internal_frame_meta_data& meta_data, const channel_settings& immutable_settings, const live_compute_options& processing)
{
	const auto phase_processing = immutable_settings.processing;
	const auto phase_retrieval = immutable_settings.retrieval;
	if (phase_processing == phase_processing::raw_frames)
	{
		return phase_retrieval_camera(meta_data, phase_retrieval);
	}
	switch (phase_retrieval)
	{
	case phase_retrieval::slim:
	case phase_retrieval::slim_demux:
	case phase_retrieval::fpm:
	case phase_retrieval::glim:
	case phase_retrieval::glim_demux:
	case phase_retrieval::polarizer_demux_single:
	case phase_retrieval::polarizer_demux_psi:
	case phase_retrieval::diffraction_phase:
	case phase_retrieval::polarizer_demux_two_frame_dpm:
		return phase_retrieval_psi(immutable_settings, meta_data, processing);
	case phase_retrieval::custom_patterns:
	case phase_retrieval::camera:
	default:
		qli_runtime_error("Not Implemented");
	}
}

camera_frame_internal compute_engine::get_pseudo_spectrum(const camera_frame_internal& phase, const bool do_pseudo_spectrum)
{
	if (phase.is_valid() && do_pseudo_spectrum)
	{
		do_pseudo_ft(*phase.buffer, phase);
	}
	return phase;
}

background_update_functors compute_engine::phase_update_functors() const
{
	background_update_functors test = { dpm_update_,slim_update,phase_update };
	return test;
}

void compute_engine::set_background_update_functors(const background_update_functors& functors)
{
	dpm_update_ = functors.dpm_functor;
	slim_update = functors.slim_functor;
	phase_update = functors.phase_functor;
}

int compute_engine::push_work(camera_frame<unsigned short> host_frame_in, const channel_settings& immutable_settings, const live_compute_options& processing)
{
	//time_slice ts("PUSH WORK");
	std::lock_guard<std::mutex> lk(protect_input_lock);
#if _DEBUG 
	if (!host_frame_in.is_valid())
	{
		qli_runtime_error("invalid frame inside queue");
	}
	if (!immutable_settings.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	const auto retrieval = immutable_settings.retrieval;
	{
		const auto is_color = (host_frame_in.samples_per_pixel == 3) || immutable_settings.demosaic == demosaic_mode::rggb_14_native;
		const auto has_color_paths = phase_retrieval_setting::settings.at(retrieval).has_color_paths;
		if (is_color && (!has_color_paths))
		{
			qli_runtime_error("Missing Color Processing Paths");
		}
	}
	//
	auto info = load_resize_and_demosaic(host_frame_in, immutable_settings, processing.is_live);
#if _DEBUG
	{
		if (info.frames_made.size() != immutable_settings.frames_per_demosaic(processing.is_live))
		{
			qli_runtime_error("Wrong number of frames made");
		}
	}
#endif
	fuzz(info);
	auto frames_made = 0;
	for (auto frame : info.frames_made)
	{
		internal_frame_meta_data meta_data(host_frame_in, host_frame_in);
		meta_data.samples_per_pixel = info.samples_per_pixel;
		meta_data.pattern_idx = frame;
		static_cast<frame_size&>(meta_data) = info.output_size;
		master_alias_check({});
		const auto phase = get_phase(meta_data, immutable_settings, processing);
#if _DEBUG
		{
			if (!phase.is_valid() && phase.buffer != nullptr)
			{
				qli_runtime_error("Stop loading garbage");
			}
		}
#endif
		master_alias_check(box(phase));
		auto phase_shifted = get_shifted_data(phase, immutable_settings);//maybe move this?
		master_alias_check(box(phase_shifted));
		const auto phase_no_bg = apply_background_and_decomplexify(phase_shifted, immutable_settings, processing);//note for DPM this de-complexifies the data!!!
#if _DEBUG
		{
			if (phase_no_bg.is_complex())
			{
				qli_runtime_error("Should have been decomplexified");
			}
		}
#endif
		master_alias_check(box(phase_no_bg));
		auto denoise_results = get_denoised_data(phase_no_bg, immutable_settings, processing.is_live);
		master_alias_check(denoise_results);
		while (!denoise_results.empty())
		{
			const auto alias_check = [&](camera_frame_internal item)
			{
				auto alias_check_temp = denoise_results;
				alias_check_temp.push_back(item);
				master_alias_check(alias_check_temp);
			};

			const auto item = denoise_results.back();
			denoise_results.pop_back();//yes this reverses the order but its easier to write like this now
			alias_check(item);
			do_filter(item, immutable_settings.retrieval, immutable_settings, immutable_settings);
			alias_check(item);
			auto pseudo_spectrum = get_pseudo_spectrum(item, immutable_settings.do_ft);
			alias_check(pseudo_spectrum);
			if (pseudo_spectrum.is_valid())
			{
				{
					std::unique_lock<std::mutex> lk(output_destination_queue_m);
#if _DEBUG
					{
						const auto some_value = pseudo_spectrum.buffer->front();
						if (!std::isfinite(some_value))
						{
							std::cout <<"Making Garbage You Suck" << std::endl;
							//qli_runtime_error("Made garbage, shouldn't happen for UX reasons");
						}
					}
#endif
					output_destination_queue.push_back(pseudo_spectrum);
					frames_made = frames_made + 1;
				}
				output_destination_queue_cv.notify_one();
			}
			master_alias_check(denoise_results);
		}
		master_alias_check({});
	}
	return frames_made;
}

void compute_engine::get_work_internal(const work_function& work, const bool is_gpu, const bool pop_front)
{
	std::unique_lock<std::mutex> lk_destination(output_destination_queue_m);
	const auto condition = [&] {return disconnect_queues_ || !output_destination_queue.empty(); };
	output_destination_queue_cv.wait(lk_destination, condition);
	if (disconnect_queues_)
	{
		return;
	}
	auto top = output_destination_queue.front();
	if (pop_front)
	{
		output_destination_queue.pop_front();
	}
	lk_destination.unlock();
	//todo rewrite
	auto& img = *top.buffer;
	if (is_gpu)
	{
		const auto ptr_d = thrust::raw_pointer_cast(img.data());
		auto frame = camera_frame<float>(ptr_d, top, top);
		work(frame);
	}
	else
	{
		thrust_safe_resize(output_copy_back_buffer, img.size());
		thrust::copy(img.begin(), img.end(), output_copy_back_buffer.begin());
		const auto ptr_h = output_copy_back_buffer.data();
		auto frame = camera_frame<float>(ptr_h, top, top);
		work(frame);
	}
	if (pop_front)
	{
		{
			std::unique_lock<std::mutex> lk_free(output_free_queue_m);
			output_free_queue.push_back(top.buffer);
		}
		output_free_queue_cv.notify_one();
	}
}

// ReSharper disable CppMemberFunctionMayBeConst
// ReSharper disable CppMemberFunctionMayBeStatic
void compute_engine::assert_that_outputs_have_been_serviced()
// ReSharper restore CppMemberFunctionMayBeStatic
// ReSharper restore CppMemberFunctionMayBeConst
{
#ifdef _DEBUG
	std::unique_lock<std::mutex> lk_destination(output_destination_queue_m);
	const auto should_be_true = output_destination_queue.empty();
	const auto should_also_be_true = frame_filtering_bank.empty();
	if (should_be_true == false)
	{
		qli_runtime_error("Logic Error");
	}
#endif
}

void compute_engine::get_work_gpu(const work_function& work, const bool pop_front)
{
	get_work_internal(work, true, pop_front);
}

void compute_engine::get_work_host(const work_function& work, const bool pop_front)
{
	get_work_internal(work, false, pop_front);
}
