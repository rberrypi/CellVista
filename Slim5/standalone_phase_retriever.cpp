#include "stdafx.h"
/*#include "compute_engine.h"
#include "write_tif.h"
#include <boost/format.hpp>

extern channel_settings get_default_settings(const processing_quad& quad);
void standalone_phase_retrieval()
{
	const std::string base_directory = R"(O:\Shares\raid6\Mikhail\GLIM_FL_Overlays\DAPI DIL SW Cells\part12_slim_redo_at_qli_lab\10x_data\fuck_it_do_by_well\well0_redo_fml\)";
	const std::string output_directory = R"(O:\Shares\raid6\Mikhail\GLIM_FL_Overlays\DAPI DIL SW Cells\part12_slim_redo_at_qli_lab\10x_data\fuck_it_do_by_well\well0_redo_fml_hilbert_output\)";
	std::unique_ptr<compute_engine> cp;
	const processing_quad quad = { phase_retrieval::slim_demux, phase_processing::phase, demosaic_mode::no_processing,denoise_mode::off};
	auto settings = get_default_settings(quad);
	settings.objective_attenuation = 4;
	settings.pixel_ratio = 1.54;
	settings.coherence_length = 2;
	static_cast<band_pass_settings&>(settings) = band_pass_settings(0.5, 250, true, true);
	for (auto c = 0; c < 9; ++c)
		for (auto r = 0; r < 10; ++r)
			for (auto z = 0; z < 3; ++z)
			{
				const roi_name name(0, 0, 0, c, r, z);
				const auto name_string = [](const roi_name& name, const auto ch, const auto m)
				{
					const auto value = (boost::format("f%d_t%d_i%d_ch%d_c%d_r%d_z%d_m%d.tif") % name.roi % name.time %name.repeat %ch %name.column %name.row %name.page %m).str();
					return value;
				};
				auto A = read_buffer<unsigned short>(base_directory + name_string(name, 0, 0));
				auto B = read_buffer<unsigned short>(base_directory + name_string(name, 0, 1));
				auto C = read_buffer<unsigned short>(base_directory + name_string(name, 0, 2));
				auto D = read_buffer<unsigned short>(base_directory + name_string(name, 0, 3));
				if (!cp)
				{
					const frame_size output_size = A;
					cp = std::make_unique< compute_engine>(output_size);
				}
				static_cast<band_pass_settings&>(settings) = band_pass_settings(0, 256, true, true);
				const auto to_camera_frame = [&](tiff_image<unsigned short>& img, const int idx)
				{
					frame_meta_data meta_data;
					meta_data.processing = quad.processing;
					meta_data.pattern_idx = idx;
					const auto data = img.img.data();
					meta_data.samples_per_pixel = img.samples_per_pixel;
					const auto frame = camera_frame<unsigned short>(data, img, meta_data);
					if (!frame.is_valid())
					{
						
					}
					return frame;
				};
				const compute_engine::work_function write_file = [&](const camera_frame<float>& frame_h)
				{
					const raw_io_work_meta_data meta_data(name, 0, { quad.retrieval, quad.processing }, false, scope_action::capture, "_hrslim");
					raw_io_work<float> io_work(frame_h, meta_data, gui_message_kind::none);
					compute_engine::write_image(io_work, output_directory);
				};
				const auto process_work = [&](const int frames_made)
				{
					for (auto i = 0; i < frames_made; ++i)
					{
						cp->get_work_host(write_file);
					}
				};
				process_work(cp->push_work(to_camera_frame(A, 0), settings, live_compute_options()));
				process_work(cp->push_work(to_camera_frame(B, 1), settings, live_compute_options()));
				process_work(cp->push_work(to_camera_frame(C, 2), settings, live_compute_options()));
				process_work(cp->push_work(to_camera_frame(D, 3), settings, live_compute_options()));
			}


}
*/
