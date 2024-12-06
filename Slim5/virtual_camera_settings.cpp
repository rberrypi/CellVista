#include "stdafx.h"
#include "virtual_camera_settings.h"
#include <type_traits>
#include "qli_runtime_error.h"

virtual_camera_settings::virtual_camera_settings_map virtual_camera_settings::settings;

camera_test_vector::camera_test_vectors camera_test_vector::tests;

[[nodiscard]] camera_test_vector virtual_camera_settings::get_camera_test_vector() const
{
	const auto is_forced_color = this->is_forced_color();
	return camera_test_vector(demosaic, retrieval, is_forced_color);
}

bool camera_test_vector::operator<(const camera_test_vector& b) const noexcept
{
	if (demosaic < b.demosaic)
	{
		return true;
	}
	if (demosaic == b.demosaic)
	{
		if (retrieval < b.retrieval)
		{
			return true;
		}
		if (retrieval == b.retrieval)
		{
			return !is_forced_color && b.is_forced_color;
		}
	}
	return false;
}

void virtual_camera_settings::transfer_settings_to_test_cameras(int slms)
{
	//todo fix these guys and check them
	//Build Test Cameras
	//todo these need to be specialized a little bit more
	constexpr auto bw_samples = 1;
	constexpr auto color_samples = 3;
	auto& settings = virtual_camera_settings::settings;
	const camera_contrast_features zyla_features(camera_chroma::monochrome, demosaic_mode::no_processing, { 0.f,65535.f });
	const camera_contrast_features old_hamamatsu(camera_chroma::monochrome, demosaic_mode::no_processing, { 0.f,4095.0f });
	const camera_contrast_features old_hamamatsu_fake_polarizer(camera_chroma::optional_polarization, demosaic_mode::polarization_0_45_90_135, { 0.f,4095.0f });
	const camera_contrast_features color_flir_features(camera_chroma::forced_color, demosaic_mode::rggb_14_native, { 0.f,65535.f });
	const camera_contrast_features polarizer_flir_features(camera_chroma::optional_polarization, demosaic_mode::polarization_0_45_90_135, { 0.f,65535.f });
	const camera_contrast_features polarizer_flir_features2(camera_chroma::optional_polarization, demosaic_mode::polarization_0_90, { 0.f,65535.f });
	const camera_contrast_features zeiss_mrc_features(camera_chroma::forced_color, demosaic_mode::no_processing, { 0.f,4095.0f });
	const processing_quad raw_processing = { phase_retrieval::camera, phase_processing::raw_frames, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad slim_bw_processing = { phase_retrieval::slim, phase_processing::raw_frames, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad dpm_bw_processing = { phase_retrieval::diffraction_phase, phase_processing::raw_frames, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad slim_rggb_processing = { phase_retrieval::slim, phase_processing::raw_frames, demosaic_mode::rggb_14_native, denoise_mode::off };
	const processing_quad slim_bw_demux_processing = { phase_retrieval::slim_demux, phase_processing::phase, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad custom_patterns_processing = { phase_retrieval::custom_patterns, phase_processing::raw_frames, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad color_calibration = { phase_retrieval::custom_patterns, phase_processing::raw_frames, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad glim_bw_processing = { phase_retrieval::glim, phase_processing::phase, demosaic_mode::no_processing, denoise_mode::off };
	//const processing_quad glim_bw_demux_processing = { phase_retrieval::glim_demux, phase_processing::phase, demosaic_mode::no_processing, denoise_mode::off };
	const processing_quad polarizer_demux_single = { phase_retrieval::polarizer_demux_single, phase_processing::raw_frames, demosaic_mode::polarization_0_45_90_135, denoise_mode::off };
	const processing_quad polarizer_demux_psi = { phase_retrieval::polarizer_demux_psi, phase_processing::raw_frames, demosaic_mode::polarization_0_45_90_135, denoise_mode::off };
	const processing_quad polarizer_demux_dpm2 = { phase_retrieval::polarizer_demux_two_frame_dpm, phase_processing::raw_frames, demosaic_mode::polarization_45_135, denoise_mode::off };
	//
	if (raw_processing.is_supported_quad())
	{
		LOGGER_INFO("virtual_camera_type::neurons_1");
		settings[virtual_camera_type::neurons_1] = [&]
		{
			const auto camera_settings = channel_settings::generate_test_channel(raw_processing, slms, bw_samples);
			return virtual_camera_settings("neurons_1", virtual_camera_settings::dpm::no, zyla_features, camera_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		LOGGER_INFO("virtual_camera_type::neurons_2");
		settings[virtual_camera_type::neurons_2] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("neurons_2", virtual_camera_settings::dpm::no, zyla_features, slim_settings);
		}();
	}
	//
	if (slim_bw_demux_processing.is_supported_quad())
	{
		LOGGER_INFO("virtual_camera_type::neurons_3");
		settings[virtual_camera_type::neurons_3] = [&]
		{
			const auto slim_demux_settings = channel_settings::generate_test_channel(slim_bw_demux_processing, slms, bw_samples);
			return virtual_camera_settings("neurons_3", virtual_camera_settings::dpm::no, zyla_features, slim_demux_settings);
		}();
	}
	//
	if (custom_patterns_processing.is_supported_quad())
	{
		settings[virtual_camera_type::calibration_ir_780nm_cropped] = [&]
		{
			const auto calibration_camera_bw = channel_settings::generate_test_channel(custom_patterns_processing, slms, bw_samples);
			return virtual_camera_settings("calibration_ir_780nm_cropped", virtual_camera_settings::dpm::no, zyla_features, calibration_camera_bw, camera_kind::calibration);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::qdic_set_1] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("qdic_set_1", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::qdic_set_2] = [&]
		{
			const auto glim_demux_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("qdic_set_2", virtual_camera_settings::dpm::no, zyla_features, glim_demux_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::qdic_rbc_set_1] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("qdic_rbc_set_1", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::rbcs_set_1] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("rbcs_set_1", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::rbcs_set_2] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("rbcs_set_2", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::segmentation_phantom] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("segmentation_phantom", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::sperm_set_1] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("sperm_set_1", virtual_camera_settings::dpm::no, zyla_features, slim_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::sperm_set_2] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("sperm_set_2", virtual_camera_settings::dpm::no, zyla_features, slim_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::sperm_set_3] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("sperm_set_3", virtual_camera_settings::dpm::no, zyla_features, slim_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::color_psi_dic_with_taps] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, color_samples);
			return virtual_camera_settings("color_psi_dic_with_taps", virtual_camera_settings::dpm::no, zeiss_mrc_features, glim_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::color_psi_phase_contrast] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, color_samples);
			return virtual_camera_settings("color_psi_phase_contrast", virtual_camera_settings::dpm::no, zeiss_mrc_features, slim_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::hela_set_1_5x] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("hela_set_1_5x", virtual_camera_settings::dpm::no, zyla_features, slim_settings);
		}();
	}
	//
	if (slim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::hela_set_2_10x] = [&]
		{
			const auto slim_settings = channel_settings::generate_test_channel(slim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("hela_set_2_10x", virtual_camera_settings::dpm::no, zyla_features, slim_settings);
		}();
	}
	//
	if (slim_rggb_processing.is_supported_quad())
	{
		settings[virtual_camera_type::mosaic_color_tissue] = [&]
		{
			const auto slim_settings_color = channel_settings::generate_test_channel(slim_rggb_processing, slms, color_samples);
			return virtual_camera_settings("mosaic_color_tissue", virtual_camera_settings::dpm::no, color_flir_features, slim_settings_color);
		}();
	}
	//
	if (color_calibration.is_supported_quad())
	{
		settings[virtual_camera_type::mosaic_color_calibration] = [&]
		{
			//duplciated because we don't have different files?
			const auto calibration_camera_color = channel_settings::generate_test_channel(color_calibration, slms, color_samples);
			return virtual_camera_settings("mosaic_color_tissue", virtual_camera_settings::dpm::no, color_flir_features, calibration_camera_color);
		}();
	};
	//
	if (polarizer_demux_single.is_supported_quad())
	{
		settings[virtual_camera_type::mosaiced_polarization_phantom] = [&]
		{
			const auto polarization_phantom = channel_settings::generate_test_channel(polarizer_demux_single, slms, bw_samples);
			return virtual_camera_settings("mosaiced_polarization_phantom", virtual_camera_settings::dpm::no, polarizer_flir_features, polarization_phantom);
		}();
	}
	//
	if (polarizer_demux_psi.is_supported_quad())
	{
		settings[virtual_camera_type::mosaiced_phantom_qdic] = [&]
		{
			const auto polarization_phantom = channel_settings::generate_test_channel(polarizer_demux_psi, slms, bw_samples);
			return virtual_camera_settings("mosaiced_phantom_qdic", virtual_camera_settings::dpm::no, polarizer_flir_features2, polarization_phantom);
		}();
	}
	//
	if (dpm_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::dpm_regular] = [&]
		{
			auto dpm_regular = channel_settings::generate_test_channel(dpm_bw_processing, slms, bw_samples);
			static_cast<dpm_settings&>(dpm_regular) = dpm_settings(498, 390, 384);
			return virtual_camera_settings("dpm_regular", virtual_camera_settings::dpm::yes, old_hamamatsu, dpm_regular);
		}();
	}
	//
	if (dpm_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::dpm_raw] = [&]
		{
			auto dpm_raw = channel_settings::generate_test_channel(dpm_bw_processing, slms, bw_samples);
			static_cast<dpm_settings&>(dpm_raw) = dpm_settings(498, 390, 384);
			return virtual_camera_settings("dpm_raw", virtual_camera_settings::dpm::yes, old_hamamatsu, dpm_raw);
		}();
	}
	//
	if (polarizer_demux_dpm2.is_supported_quad())
	{
		settings[virtual_camera_type::dpm_polarization_psi] = [&]
		{
			const auto polarization_phantom2 = channel_settings::generate_test_channel(polarizer_demux_dpm2, slms, bw_samples);
			return virtual_camera_settings("dpm_polarization_psi", virtual_camera_settings::dpm::yes, old_hamamatsu_fake_polarizer, polarization_phantom2);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::qdic_dapi_set_1] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("qdic_dapi_set_1", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//
	if (glim_bw_processing.is_supported_quad())
	{
		settings[virtual_camera_type::sw480_10x_slim_ml_set] = [&]
		{
			const auto glim_settings = channel_settings::generate_test_channel(glim_bw_processing, slms, bw_samples);
			return virtual_camera_settings("sw480_10x_slim_ml_set", virtual_camera_settings::dpm::no, zyla_features, glim_settings);
		}();
	}
	//Check all file names
	for (const auto& item : settings)
	{
		item.second.bool_verify_resource_path();
	}

	LOGGER_INFO("settings.size(): " << settings.size());


	//limit our tests to only certain phantoms
	for (auto camera : {
		virtual_camera_type::neurons_1//, virtual_camera_type::neurons_2, virtual_camera_type::neurons_3,// SLIM BW Test Cameras
		/*
		virtual_camera_type::qdic_set_1, virtual_camera_type::qdic_set_2, // GLIM Test Cameras
		virtual_camera_type::dpm_regular, // DPM
		virtual_camera_type::mosaic_color_tissue, // Mosaiced colors
		virtual_camera_type::color_psi_phase_contrast, // Forced color tests
		virtual_camera_type::mosaiced_polarization_phantom,virtual_camera_type::mosaiced_phantom_qdic,  // Polarization Phantoms
		virtual_camera_type::tissue_1,virtual_camera_type::mosaic_color_calibration //Calibration Cameras
		*/
		})
	{
		const auto test_vector = virtual_camera_settings::settings.at(camera).get_camera_test_vector();
		if (camera_test_vector::tests.find(test_vector) != camera_test_vector::tests.end())
		{
			qli_runtime_error("Test Vectors Miss Configured");
		}
		camera_test_vector::tests[test_vector] = camera;
	}
}
