#include "stdafx.h"
#if INCLUDE_ML!=0
#include "ml_transformer.h"
#include <thrust/device_vector.h>
#include "write_tif.h"
#include "thrust_resize.h"
#include "write_debug_gpu.h"
#include <numeric>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include <boost/format.hpp>
#include <mutex>
auto make_range(int start, int stop) {
	std::vector<int> range(stop - start);
	std::iota(range.begin(), range.end(), start);
	return range;
};

struct work_item
{
	std::experimental::filesystem::path input, output;
	ml_remapper_file::ml_remapper_types type;
};

void standalone_ml_processor()
{
	std::vector<work_item> items;
	/*
	auto work_items = [&] {
		const auto input_directory = std::string(R"(O:\Shares\raid6\Mikhail\GLIM_FL_Overlays\DAPI DIL SW Cells\part4 timelapse only post stain\SW 480 and 620 multiwell _redo_fl_scan_longer_fo_fun_hilbert\)");
		const auto outputs = {
			std::make_pair( ml_remapper_file::ml_remapper_types::glim_dapi_20x,std::string(R"(O:\Shares\raid6\Mikhail\Digital Staining\Figure 3\20x_glim_inference\dapi)")),
			std::make_pair(ml_remapper_file::ml_remapper_types::glim_dil_20x,std::string(R"(O:\Shares\raid6\Mikhail\Digital Staining\Figure 3\20x_glim_inference\dil)")),
		};
		const auto input_name = "hilbert";
		auto fovs = make_range(0, 2);
		auto times = make_range(0, 1);
		auto iterations = make_range(0, 1);
		auto channels = make_range(0 + 2, 1 + 2);
		auto columns = make_range(0, 7);
		auto rows = make_range(0, 7);
		auto pages = make_range(6, 10);
		//f0_t0_i0_ch2_c0_r0_z0_mhilbert.tif
		const auto input_directory_std = std::experimental::filesystem::path(input_directory);
		for (const auto& output : outputs)
		{
			const auto output_directory_std = std::experimental::filesystem::path(output.second);
			const auto type = output.first;
			for (auto f : fovs)
			{
				for (auto t : times)
				{
					for (auto i : iterations)
					{
						for (auto ch : channels)
						{
							for (auto c : columns)
							{
								for (auto r : rows)
								{
									for (auto z : pages)
									{
										//f0_t0_i0_ch2_c0_r0_z0_mhilbert.tif
										auto input_file = [&]() {
											const auto input_file_path = boost::str(boost::format("f%d_t%d_i%d_ch%d_c%d_r%d_z%d_m%s.tif") % f% t% i% ch% c% r% z% input_name);
											const auto input_file_path_std = std::experimental::filesystem::path(input_file_path);
											return input_directory_std / (input_file_path_std);
										}();
										auto output_file = [&]() {
											const auto output_file_path = boost::str(boost::format("f%d_t%d_i%d_ch%d_c%d_r%d_z%d_m%s.tif") % f% t% i% ch% c% r% z% input_name);
											const auto output_file_path_std = std::experimental::filesystem::path(output_file_path);
											return output_directory_std / (output_file_path_std);
										}();
										const auto input_exists = std::experimental::filesystem::exists(input_file);
										const auto output_exists = std::experimental::filesystem::exists(output_file);

										if (input_exists && !output_exists)
										{
											items.push_back({ input_file,output_file ,type });
										}
										if (!input_exists)
										{
											std::cout << input_file.string() << std::endl;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		escape:
		return items;
	}();
	*/

	const auto input_filename = R"(C:\Users\Misha\Desktop\u-net_tensorrt-master\Project2\pure_glim\f0_t0_i0_ch0_c0_r2_z4_mhilbert.tif)";
	const auto output_filename_dapi = R"(C:\Users\Misha\Desktop\u-net_tensorrt-master\Project2\pure_glim\f0_t0_i0_ch0_c0_r2_z4_mhilbert_output.tif)";
	const work_item item_dapi = { input_filename,output_filename_dapi, ml_remapper_file::ml_remapper_types::glim_dapi_20x_480 };
	items.push_back(item_dapi);
	ml_transformer ml;
	const auto thread_count = 1;
	std::mutex protect_text_output, protect_ml;
	std::array< thrust::device_vector<float>, thread_count> input_d_array;
	std::array< thrust::device_vector<float>, thread_count> output_d_array;
	std::mutex screw_it; // ask romanian fuck cunt to fix it
	{
		const auto do_work_item = [&](const work_item& item, int thread_idx)
		{
			const auto start = timestamp();

			const auto input = item.input.string();
			const auto output = item.output.string();
			auto input_h = read_buffer<float>(input);
			{
				std::unique_lock<std::mutex> lk(screw_it);
				ml.set_network_size(input_h);
			}
			const ml_remapper_file info = ml_remapper_file::ml_remappers.at(item.type);
			auto& input_d = input_d_array.at(thread_idx);
			auto& output_d = output_d_array.at(thread_idx);
			float* input_ptr = thrust_safe_get_pointer(input_d, input_h.n());
			CUDASAFECALL(cudaMemcpy(input_ptr, input_h.img.data(), input_h.bytes(), cudaMemcpyKind::cudaMemcpyHostToDevice));
			const ml_remapper remapper(item.type, { 0,1 }, 1, ml_remapper::display_mode::overlay);
			const frame_size output_size = ml_transformer::get_ml_output_size(input_h, info.designed_pixel_ratio, item.type);
			float* output_ptr = thrust_safe_get_pointer(output_d, output_size.n());
			{
				std::unique_lock<std::mutex> lk(protect_ml);
				ml.do_ml_transform(output_ptr, output_size, input_ptr, input_h, remapper, info.designed_pixel_ratio, false);
			}
			write_debug_gpu(output_ptr, output_size.width, output_size.height, 1, output.c_str(), true);
			{
				std::unique_lock<std::mutex> lk(protect_text_output);
				const auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>((timestamp() - start)).count();
				static auto running_runtime = runtime;
				running_runtime = 0.95f * running_runtime + 0.05f * runtime;
				std::cout << "Time Between " << static_cast<int>(running_runtime / thread_count) << ": " << runtime << " " << output << std::endl;
			}
		};
		std::mutex protect_queue;
		auto work_function = [&](auto thread_id) {
			while (true)
			{
				std::unique_lock < std::mutex> lk(protect_queue);
				if (items.empty())
				{
					return;
				}
				const auto work_item = items.back();
				items.pop_back();
				lk.unlock();
				do_work_item(work_item, thread_id);
			}
		};
		std::vector<std::thread> work_threads;
		for (auto i = 0; i < thread_count; ++i)
		{
			work_threads.emplace_back(std::thread(work_function, i));
		}
		for (auto&& t : work_threads)
		{
			t.join();
		}
	}

}
#endif