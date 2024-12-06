#include "stdafx.h"
#include "device_factory.h"
#include "acquisition_framework.h"
#include "scope.h"
#include "slm_device.h"
#include "remake_directory.h"
#include <iostream>
#include "render_engine_pass_through.h"
#include "camera_device.h"
#include "trakem2_stitching_structs.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include "thrust_resize.h"
#include "virtual_camera_device.h"
#include "virtual_camera_settings.h"
#include "write_tif.h"
#include "compute_engine.h"
//const auto test_modes = { program_mode::sync_capture_sync_io, program_mode::sync_capture_async_io, program_mode::async_capture_async_io };
const auto test_modes = { capture_mode::sync_capture_sync_io };

extern std::string make_slm_pattern(int pattern);
std::shared_ptr<compute_engine> testing_engine;
[[nodiscard]] auto run_test(const acquisition& test, const capture_mode capture_mode, const bool console = true)
{
	remake_directory(test.output_dir);
	acquisition_framework wrk(testing_engine);
	D->route = test;
	D->io_show_cmd_progress = console;
	const auto max_frame_size = D->max_camera_frame_size();
	auto engine_pass_through = std::make_unique<render_engine_pass_through>(max_frame_size);
	auto result = wrk.capture_wrapper(capture_mode, engine_pass_through.get());
	return result;
}

struct test_vector_results final
{
	size_t accumulated_file_size;
	int count;
	bool item_is_garbage;
	image_info sample_info;
	test_vector_results() noexcept : accumulated_file_size(0), count(0), item_is_garbage(false) {}
	[[nodiscard]] size_t size_per_file() const noexcept
	{
		return accumulated_file_size / count;
	}
};

enum class count_type { sixteen, single, integer };
test_vector_results count_files(const std::string& dir, const std::string& suffix, const count_type type)
{

	test_vector_results count;
	const auto folder_path = std::experimental::filesystem::path(dir);
	std::experimental::filesystem::path example_item;
	for (auto it = std::experimental::filesystem::directory_iterator(folder_path); it != std::experimental::filesystem::directory_iterator(); ++it)
	{
		auto path = it->path();
		const auto extension = path.extension();
		if (extension == suffix)
		{
			example_item = it->path();
			count.accumulated_file_size = count.accumulated_file_size + std::experimental::filesystem::file_size(*it);
			count.count = count.count + 1;
		}
	}
	count.item_is_garbage = [&] {
		const auto absolute = std::experimental::filesystem::absolute(example_item).string();
		switch (type)
		{
		case count_type::sixteen:
		{
			const auto sixteen = read_buffer<unsigned short>(absolute);
			count.sample_info = static_cast<const image_info&>(sixteen);
			return sixteen.is_garbage();
		}
		case count_type::single:
		{
			const auto single = read_buffer<float>(absolute);
			count.sample_info = static_cast<const image_info&>(single);
			return single.is_garbage();
		}
		default:
			qli_not_implemented();
		}
	}();
	return count;
}

[[nodiscard]] std::string get_quad_test_label(const camera_chroma chroma, const processing_quad& quad)
{
	const auto& chroma_name = camera_chroma_setting::settings.at(chroma).name;
	const auto& demosaic_name = demosaic_setting::info.at(quad.demosaic).label;
	const auto& retrieval_name = phase_retrieval_setting::settings.at(quad.retrieval).label;
	const auto& processing_name = phase_processing_setting::settings.at(quad.processing).label;
	const auto& denoise_name = denoise_setting::settings.at(quad.denoise).label;
	return chroma_name + "_" + demosaic_name + "_" + retrieval_name + "_" + processing_name + "_" + denoise_name;
}

std::vector<camera_test_vector> get_vectors()
{
	// std::vector<camera_test_vector> test_vectors;
	//auto phase_retrievals = { phase_retrieval::camera, phase_retrieval::custom_patterns,  phase_retrieval::slim , phase_retrieval::slim_demux, phase_retrieval::glim, phase_retrieval::glim_demux, phase_retrieval::diffraction_phase, phase_retrieval::polarizer_demux_single, phase_retrieval::polarizer_demux_psi, phase_retrieval::polarizer_demux_two_frame_dpm };
	// auto phase_retrievals = { phase_retrieval::glim_demux };
	// auto demosaic_modes = { demosaic_mode::no_processing, demosaic_mode::rggb_14_native, demosaic_mode::polarization_0_45_90_135, demosaic_mode::polarization_0_90, demosaic_mode::polarization_45_135 };
	// for (auto forced_color : { true,false })
	// {
	// 	for (auto demosaic : demosaic_modes)
	// 	{
	// 		for (auto retrieval : phase_retrievals)
	// 		{
	// 			const auto test_vector = camera_test_vector(demosaic, retrieval, forced_color);
	// 			test_vectors.push_back(test_vector);
	// 		}
	// 	}
	// }
	std::vector<camera_test_vector> test_vectors = { camera_test_vector(demosaic_mode::no_processing, phase_retrieval::custom_patterns, false) };
	return test_vectors;
}

void all_capture_modalities(const std::string& basedir)
{
	{
		//dirty hack to initialize list of virtual cameras so virtual feature is populated
		D = std::make_unique<device_factory>(virtual_camera_type::neurons_1);
	}
	// ReSharper disable CppInitializedValueIsAlwaysRewritten
	auto capture_modes = { capture_mode::sync_capture_sync_io };
	auto test_vector = get_vectors();
	if constexpr (false)
	{
		const auto  resume_vector = processing_quad(phase_retrieval::camera, phase_processing::phase, demosaic_mode::no_processing, denoise_mode::off);
		auto start = std::find(test_vector.begin(), test_vector.end(), resume_vector);
		test_vector.erase(test_vector.begin(), start);
	}
	const auto get_memory = []
	{
		const auto info = get_cuda_memory_info();
		return (info.total_byte - info.free_byte) / static_cast<float>(info.total_byte);
	};
	//const auto denoise_mode_test = { denoise_mode::off,denoise_mode::hybrid };
	const auto denoise_mode_test = { denoise_mode::off };
	std::vector<float> memory_history;
	for (const auto capture_mode : capture_modes)
	{
		for (const auto test : test_vector)
		{
			if (camera_test_vector::tests.find(test) == camera_test_vector::tests.end())
			{
				continue;
			}
			const auto camera_type = camera_test_vector::tests.at(test);
			for (const auto denoise_mode : denoise_mode_test)
			{
				const auto& processing_modes = phase_retrieval_setting::settings.at(test.retrieval).supported_processing_modes;
				for (const auto processing_mode : processing_modes)
				{
					const auto skip_mode = phase_processing_setting::settings.at(processing_mode).skip_when_testing;
					if (skip_mode)
					{
						continue;
					}
					const auto quad = processing_quad(test.retrieval, processing_mode, test.demosaic, denoise_mode);
					if (!quad.is_supported_quad())
					{
						qli_runtime_error("Don't Do This Test");
					}
					{
						const auto capture_mode_name = capture_mode_settings::info.at(capture_mode).name;
						auto channel_settings = virtual_camera_settings::settings.at(camera_type);
						const auto label = get_quad_test_label(channel_settings.chroma, quad);
						const auto directory = std::string(basedir).append(capture_mode_name).append("_").append(label);
						if (std::filesystem::exists(std::filesystem::path(directory)))
						{
							continue;//skip passed tests
						}
						memory_history.push_back(get_memory());
						static_cast<processing_quad&>(channel_settings) = quad;
						channel_settings.assert_validity();
						const auto is_dpm = channel_settings.is_dpm();
						if (is_dpm && !channel_settings.dpm_phase_is_complete())
						{
							qli_runtime_error("DPM Settings Invalid");
						}
						D = std::make_unique<device_factory>(camera_type);
						testing_engine = std::make_shared<compute_engine>(D->max_camera_frame_size());
						thrust_enforce_no_allocation = true;
						//settings.bin_index = 1;
						acquisition experiment;
						experiment.ch.push_back(channel_settings);
						constexpr auto snapshots = 5;
						const auto loc = static_cast<scope_location_xyz>(D->scope->get_state());
						if (is_dpm)
						{
							//add background image
							roi_name roi;
							experiment.cap.emplace_back(capture_item(roi, scope_delays(), loc, 0, false, scope_action::set_bg_for_this_channel));
						}
						for (auto i = 0; i < snapshots; i++)
						{
							roi_name roi;
							roi.time = i;
							experiment.cap.emplace_back(capture_item(roi, scope_delays(), loc, 0));
						}
						experiment.output_dir = directory;
						constexpr auto is_live = false;
						const auto expected_patterns = channel_settings.output_files_per_compute(is_live);
						const auto expected_size_per_item = static_cast<double>(channel_settings.bytes_per_capture_item_on_disk());
						run_test(experiment, capture_mode);
						const auto expected_files = snapshots * expected_patterns;
						const auto count_type = channel_settings.is_native_sixteen_bit() ? count_type::sixteen : count_type::single;
						const auto stats = count_files(directory, ".tif", count_type);
						const auto received_frames = stats.count;
						std::cout << "Test : " << directory << std::endl;
						if (received_frames != expected_files || received_frames == 0)
						{
							const auto* error_message = "Some Files Are Missing!";
							qli_runtime_error(error_message);//Also require manual inspection!!!
						}
						{
							const auto expected_info = channel_settings.image_info_per_capture_item_on_disk();
							if (stats.sample_info != expected_info)
							{
								const auto* error_str = "Size Mismatch";
								qli_runtime_error(error_str);
							}
						}
						{
							const auto size_per_item = static_cast<double>(stats.size_per_file());
							constexpr auto overhead_bytes = 8575;//1471 bytes overhead
							const auto overshoot = (size_per_item - (expected_size_per_item + overhead_bytes)) / (expected_size_per_item + overhead_bytes);
							//const auto supposed_overhead = size_per_item-expected_size_per_item;
							if (abs(overshoot) > 0.2)
							{
								const auto* error_str = "File Size Mismatch";
								qli_runtime_error(error_str);
							}
						}
						{
							if (stats.item_is_garbage)
							{
								const auto* error_str = "Processing Error";
								qli_runtime_error(error_str);
							}
						}

						std::cout << "Test Complete" << std::endl;
						thrust_enforce_no_allocation = false;
					}
				}

			}
		}
		// ReSharper restore CppInitializedValueIsAlwaysRewritten
	}
}

void ml_speed_tests(const std::string& basedir)
{
	qli_not_implemented();
	/*
	const auto capture_mode = program_mode::sync_capture_sync_io;
	auto settings = get_default_settings({ phase_retrieval::slim_demux,phase_processing::phase,demosaic_mode::no_processing,denoise_mode::off });
	static_cast<band_pass_settings&>(settings) = band_pass_settings(0, 250, true, true);
	//auto settings = get_default_settings({ phase_retrieval::glim_demux,phase_processing::phase,demosaic_mode::no_processing }, denoise_mode::off);
	settings.pixel_ratio = 1.57;
	settings.qsb_qdic_shear_dx = 1;
	//settings.qsb_qdic_shear_angle = 57;
	settings.qsb_qdic_shear_angle = 45;
	settings.frames.at(0).weights.at(0).top = 0.1357000023126602;
	settings.frames.at(0).weights.at(0).bot = 0.48399999737739565;
	settings.frames.at(0).weights.at(0).constant = 0.252600014209747;

	settings.frames.at(1).weights.at(0).top = -0.4796000123023987;
	settings.frames.at(1).weights.at(0).bot = 0.13449999690055848;
	settings.frames.at(1).weights.at(0).constant = 0.24809999763965608;

	settings.frames.at(2).weights.at(0).top = -0.1378999948501587;
	settings.frames.at(2).weights.at(0).bot = -0.47780001163482668;
	settings.frames.at(2).weights.at(0).constant = 0.24740000069141389;

	settings.frames.at(3).weights.at(0).top = 0.48179998993873598;
	settings.frames.at(3).weights.at(0).bot = -0.14059999585151673;
	settings.frames.at(3).weights.at(0).constant = 0.25189998745918276;

	channel_settings::bake_in_ml_render_test(settings);

	settings.assert_validity();
	acquisition experiment;
	experiment.ch.push_back(settings);
	const auto snapshots = 1;
	const auto loc = static_cast<scope_location_xyz>(D->scope->get_state());
	for (auto i = 0; i < snapshots; i++)
	{
		roi_name roi;
		roi.time = i;
		experiment.cap.emplace_back(capture_item(roi, scope_delays(), loc, 0));
	}
	*/
}

void trakem2_tests(const std::string& basedir)
{
	//Part 1 make the directory
	const auto directory = std::string(basedir).append("TrakemTest");
	remake_directory(directory);
	const processing_quad processing_quad(phase_retrieval::slim, phase_processing::phase, demosaic_mode::no_processing, denoise_mode::off);
	auto base_settings = channel_settings::generate_test_channel(processing_quad, 1, 1);
	base_settings.aoi_index = 1;
	//Part 2 get calibration acquisition
	constexpr auto x_calibration_step_um = 125.f;
	constexpr auto y_calibration_step_um = 230.f;
	acquisition calibration_acquisition;
	calibration_acquisition.ch.push_back(base_settings);
	{
		for (auto r = 0; r < 2; ++r)
		{
			for (auto c = 0; c < 2; ++c)
			{
				const roi_name roi_name(0, 0, 0, c, r, 0);
				const scope_location_xyz location(x_calibration_step_um * c, y_calibration_step_um * r, 0);
				const capture_item an_item(roi_name, scope_delays(), location, 0);
				calibration_acquisition.cap.push_back(an_item);
			}
		}
	}

	//Part 3 write test 
	constexpr auto calibrated_pixel_ratio = 2.35f;
	const auto default_mapper = trakem2_stage_coordinate_to_pixel_mapper::get_pass_through_mapper(calibrated_pixel_ratio);
	const auto calibration = calibration_info({ x_calibration_step_um ,y_calibration_step_um,calibrated_pixel_ratio });
	auto calibration_acquisition_calibration = trakem2_processor::acquisition_to_trakem2(calibration_acquisition, default_mapper, calibrated_pixel_ratio, calibration);
	static_cast<calibration_info&>(calibration_acquisition_calibration.trakem2_files.begin()->second) = calibration;
	const auto right_number_of_calibration_files = calibration_acquisition_calibration.trakem2_files.begin()->second.t2_layers.size() == 1;
	const auto right_number_of_tiles = calibration_acquisition_calibration.trakem2_files.begin()->second.t2_layers.begin()->t2_patch_list.size() == 4;
	if (!(right_number_of_calibration_files && right_number_of_tiles))
	{
		qli_runtime_error("Invalid Acquisition");
	}
	trakem2_processor::write_trakem2(calibration_acquisition_calibration, directory);

	//Part 4 from the test get the settings (these should match)
	const auto alignment_path = [&]
	{
		const auto& file = calibration_acquisition_calibration.trakem2_files.begin();
		const auto file_stub = file->second.get_filename(file->first);
		return std::string(directory).append("\\").append(file_stub);
	}();
	const auto calibration_vector_measured = trakem2_processor::get_vectors_from_xml_file(alignment_path);
	if (!calibration_vector_measured.is_valid())
	{
		qli_runtime_error("Oh Nope");
	}
	const auto microns_to_pixel_mapper = trakem2_stage_coordinate_to_pixel_mapper(calibration_vector_measured);

	//Part 5 write the acquisition file
	acquisition evaluation_acquisition;
	//const auto ri_max = 2,chi_max = 3, ti_max = 4, ii_max = 5, zi_max = 6, xi_max=7, yi_max=8 ;
	constexpr auto ri_max = 1, chi_max = 1, ti_max = 3, ii_max = 1, zi_max = 1, xi_max = 3, yi_max = 3;
	constexpr auto expected_files = ri_max * chi_max * ii_max * zi_max;
	constexpr auto expected_layers = ti_max;
	constexpr auto expected_tiles_per_layer = xi_max * yi_max;
	constexpr auto x_step = 100;
	constexpr auto y_step = 200;
	for (auto chi = 0; chi < chi_max; ++chi)
	{
		evaluation_acquisition.ch.push_back(base_settings);
		for (auto ti = 0; ti < ti_max; ++ti)
		{
			for (auto ri = 0; ri < ri_max; ++ri)
			{
				for (auto ii = 0; ii < ii_max; ++ii)
				{
					for (auto xi = 0; xi < xi_max; ++xi)
					{
						for (auto yi = 0; yi < yi_max; ++yi)
						{
							for (auto zi = 0; zi < zi_max; ++zi)
							{
								const roi_name roi_name(ri, ti, ii, xi, yi, zi);
								const scope_location_xyz location(xi * x_step, yi * y_step, zi * 12);
								const capture_item an_item(roi_name, scope_delays(), location, chi);
								evaluation_acquisition.cap.push_back(an_item);
							}
						}
					}
				}
			}
		}
	}
	evaluation_acquisition.ch.push_back(base_settings);
	const auto current_pixel_ratio = 2 * calibrated_pixel_ratio;
	const auto evaluation_acquisition_trakem2 = trakem2_processor::acquisition_to_trakem2(evaluation_acquisition, microns_to_pixel_mapper, current_pixel_ratio);
	trakem2_processor::write_trakem2(evaluation_acquisition_trakem2, directory);
	{
		const auto generated_files = evaluation_acquisition_trakem2.trakem2_files.size();
		const auto right_number_of_files = generated_files == expected_files;
		const auto& first_layers = evaluation_acquisition_trakem2.trakem2_files.begin()->second.t2_layers;
		const auto right_number_of_layers = first_layers.size() == expected_layers;
		const auto& first_layer = first_layers.front();
		const auto right_number_of_tiles2 = first_layer.t2_patch_list.size() == expected_tiles_per_layer;
		if (!(right_number_of_files && right_number_of_layers && right_number_of_tiles2))
		{
			qli_runtime_error("Test Failed, Good Work Team");
		}
	}

}

void burst_tests(const std::string& basedir)
{
	D = std::make_unique<device_factory>(virtual_camera_type::neurons_1);
	testing_engine = std::make_shared<compute_engine>(D->max_camera_frame_size());
	const auto loc = static_cast<scope_location_xyz>(D->scope->get_state());
	acquisition experiment;
	const auto directory = std::string(basedir).append("_Burst");
	experiment.output_dir = directory;
	const auto channel = channel_settings::generate_test_channel({ phase_retrieval::camera,phase_processing::raw_frames,demosaic_mode::no_processing,denoise_mode::off });
	for (const auto& camera : D->cameras)
	{
		if (camera->has_burst_mode)
		{
			for (auto i = 0; i < 21; ++i)
			{
				const auto roi = roi_name(0, 0, i, 0, 0, 0);
				experiment.cap.emplace_back(capture_item(roi, scope_delays(), loc, experiment.ch.size()));

			}
			experiment.ch.push_back(channel);
		}
	}
	const auto acquired_frames = run_test(experiment, capture_mode::burst_capture_async_io);
}

void system_tests()
{
	const auto* base_dir = "C:\\tests\\";
	//
	std::vector<std::function<void(std::string)>> tests = { all_capture_modalities };
	for (size_t i = 0; i < tests.size(); ++i)
	{
		std::cout << "Running Test " << i << "/" << tests.size() << std::endl;
		const auto test = tests.at(i);
		test(base_dir);
	}
	CUDASAFECALL(cudaDeviceReset());
}
