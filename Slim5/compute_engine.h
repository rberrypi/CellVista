#pragma once
#ifndef COMPUTE_ENGINE_H
#define COMPUTE_ENGINE_H
#include "dpm_gpu_structs.h"
#include "slim_gpu_structs.h"
#include "other_patterns.h"
#include "haar_wavelet_gpu.h"
#include "npp_histogram.h"
#include "pseudo_spectrum.h" 
#include "fourier_filter.h"
#include "compute_engine_demosaic.h"
#include <queue>
#include <mutex>
#include "write_tif.h"
#include "polarization_filters.h"
#include "io_work.h"
#include "qli_runtime_error.h"
#include <itaSettings.h>
#include <iostream>

using std::cout;
using std::endl;

struct live_compute_options;
template<typename T>
struct three_element_filler final
{
	T item[3];
};

class compute_engine  final : public dpm_gpu_structs, slim_gpu_structs, other_patterns, public haar_wavelet_gpu, public cuda_npp, pseudo_spectrum, fourier_filter, demosaic_structs, polarization_filters
{
	//the output queue is unused and should be eliminated
	camera_frame_internal get_phase(const internal_frame_meta_data& meta_data, const channel_settings& immutable_settings, const live_compute_options& processing);
	camera_frame_internal apply_background_and_decomplexify(camera_frame_internal& frame_internal, const channel_settings& settings, const live_compute_options& processing_options);
	typedef boost::container::small_vector<camera_frame_internal, 5> denoise_output_package;
	denoise_output_package get_denoised_data(const camera_frame_internal& phase, const channel_settings& settings, bool is_live);
	camera_frame_internal get_shifted_data(const camera_frame_internal& phase, const render_shifter& shifter);
	camera_frame_internal get_pseudo_spectrum(const camera_frame_internal& phase, bool do_pseudo_spectrum);
	void master_alias_check(const denoise_output_package& frames) const;
	static denoise_output_package box(const camera_frame_internal& input)
	{
		denoise_output_package result(1);
		result.front() = input;
		return result;
	}
	camera_frame_internal_buffer* get_out_frame();
	//
	phase_update_functor phase_update;
	//
	//
public:
	compute_engine(const compute_engine&) = delete;
	explicit compute_engine(const frame_size& output_size);
	void kill_queues();
	virtual ~compute_engine();
	//
	//always allocate for SLIM?, maybe in static use RAII
	typedef std::function<void(camera_frame<float>&)> work_function;
	void set_background_update_functors(const background_update_functors& functors);
	[[nodsicard]] background_update_functors phase_update_functors() const;

	camera_frame_internal phase_retrieval_camera(const internal_frame_meta_data& meta_data, phase_retrieval retrieval);
	camera_frame_internal phase_retrieval_psi(const channel_settings& settings, const internal_frame_meta_data& meta_data, const live_compute_options& processing_options);
	frame_size phase_retrieval_pol(out_frame output_frame, const channel_settings& settings, const internal_frame_meta_data& meta_data, const live_compute_options& processing_options);
	int push_work(camera_frame<unsigned short> host_frame_in, const channel_settings& immutable_settings, const live_compute_options& processing);//pushes work, stores result in output buffer
	void get_work_internal(const work_function& work, bool is_gpu, bool pop_front);//Also locking
	void get_work_gpu(const work_function& work, bool pop_front = true);//Also locking
	void get_work_host(const work_function& work, bool pop_front = true);//Also locking
	
																		 
	//should be forward declared to prevent leaking the write_tif include
	template<typename T>
	void static write_image(raw_io_work<T>& in, const std::experimental::filesystem::path& dir)
	{
		if (in.force_sixteen_bit)
		{
			const auto number_of_elements = in.n() * in.samples_per_pixel;
			static thread_local  std::vector<unsigned short> sixteen_bit_conversion_buffer(number_of_elements);
			sixteen_bit_conversion_buffer.resize(number_of_elements);
			auto ptr_sixteen = sixteen_bit_conversion_buffer.data();
#pragma warning(disable:4244)  
			std::copy(in.img, in.img + number_of_elements, sixteen_bit_conversion_buffer.begin());
#pragma warning(default:4244) 
			camera_frame<unsigned short> frame_sixteen(ptr_sixteen, in, in);
			raw_io_work<unsigned short> short_work(frame_sixteen, in, in.gui_message);
			short_work.force_sixteen_bit = false;//<-else stack overflow yo
			write_image(short_work, dir);
			return;
		}
		if ((!std::is_same<T, unsigned short>::value) && in.force_sixteen_bit)
		{
		}
		const auto name = in.get_path(file_kind::image, dir);
		raw_io_work_meta_data::ensure_directories_exist(name);
		const auto string = name.u8string();

		// Add code to call the model for ITA here, string is the location of the file
		// send signal to the ITA process here
		bool write = true;
#if FUCKFACEGABRIELPOPESCU
		{
			write = false;
			switch (itaSettings::current_trigger_condition)
			{
			case 0:
				LOGGER_INFO("No trigger condition set, fuck popescu.");
				write = true;
				break;
			case 1:
				LOGGER_INFO("Running live dead ratio trigger");
				write = itaSettings::live_dead_trigger(in.img, in.width, in.height, in.samples_per_pixel);
				break;
			case 2:
				LOGGER_INFO("Cell count Trigger");
				write = itaSettings::cell_count_trigger(in.img, in.width, in.height, in.samples_per_pixel);
				break;
			case 3:
				LOGGER_INFO("Confluency trigger");
				write = itaSettings::confluency_trigger(in.img, in.width, in.height, in.samples_per_pixel);
				break;
			default:
				write = true;
				LOGGER_WARN("Unknown trigger condition, now go ask Gabriel to shove his head up his ass like a fucking ostrich.");
				break;
			}
		}
#endif
		if (write) {
			write_tif(string, in.img, in.width, in.height, in.samples_per_pixel, nullptr);
			LOGGER_INFO("saving image: " << string);
		}
	}
	//note your going to 
	std::vector<unsigned short> bit_convert_buffer;
	//Filters
	static void move_clamp_and_scale(unsigned char* out_d_8_bit_img, const  float* img_d, const frame_size& frame_size, int samples_per_pixel, const display_settings::display_ranges& range);//todo better name
	static void get_five_tap_filter(camera_frame_internal_buffer& res, const std::deque<camera_frame_internal>& input_frames, denoise_mode filter);
	static void get_hybrid_filter(camera_frame_internal_buffer& res, const std::deque<camera_frame_internal>&);
	//
	std::vector<camera_frame_internal_buffer> output_frames;
	//
	std::mutex output_free_queue_m;
	std::condition_variable output_free_queue_cv;
	std::deque<camera_frame_internal_buffer*> output_free_queue;
	//
	std::mutex output_destination_queue_m;
	std::condition_variable output_destination_queue_cv;
	std::deque<camera_frame_internal> output_destination_queue;
	//
	bool disconnect_queues_;
	thrust::host_vector<float> output_copy_back_buffer;
	thrust::device_vector<float> output_copy_to_render_buffer;
	//
	std::deque<camera_frame_internal> frame_filtering_bank;
	camera_frame_internal_buffer filter_buffer;
	camera_frame_internal_buffer* filter_buffer_ptr;
	camera_frame_internal_buffer shifter_buffer;
	camera_frame_internal_buffer* shifter_buffer_ptr;
	camera_frame_internal_buffer decomplexify_buffer;
	camera_frame_internal_buffer* decomplexify_buffer_ptr;
	//
	void flush_and_reset();
	void assert_that_outputs_have_been_serviced();

};

#endif

