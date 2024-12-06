#include "virtual_camera_device.h"
#include "write_tif.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <npp.h>
#include "npp_error_check.h"
#include "time_slice.h"
#include "cuda_error_check.h"
#include <QDebug>
#include "thrust_resize.h"
#include "virtual_camera_settings.h"
#include "virtual_camera_device.h"
#include <QTemporaryFile>
#include "write_tif.h"
#include <iostream>
#include <sstream>
#include <algorithm>

tiff_image<unsigned short> read_sixteen_bit_resource(const QString& resource, bool skip_read)
{
	static bool success = virtual_camera_device::register_resource();
	QFile file(resource);
	if (!file.exists())
	{
		std::stringstream error_msg;
		error_msg << "Failed to find: " << file.fileName().toStdString();
		qli_runtime_error(error_msg.str());
	}
	if (skip_read)
	{
		return tiff_image<unsigned short>();
	}
	const auto temporary_file = QTemporaryFile::createNativeFile(file);
	const auto full_path = temporary_file->fileName();
	const auto full_path_string = full_path.toStdString();
	return read_buffer<unsigned short>(full_path_string);
}

template<typename T>
thrust::device_vector<T> load_tiff_img(const tiff_image<T>& input)
{
	const auto samples = input.samples();
	thrust::device_vector<T> pseudo_frame(samples);
	if (samples > 0)
	{

		const auto ptr = raw_pointer_cast(pseudo_frame.data());
		CUDASAFECALL(cudaMemcpy(ptr, input.img.data(), samples * sizeof(T), cudaMemcpyHostToDevice));
	}
	return pseudo_frame;
}

tiff_image<unsigned short> read_frame(const virtual_camera_settings& info, int pattern, bool is_foreground, bool skip_read)
{
	const auto is_dpm = info.is_dpm();
	const auto post_fix = is_dpm ? (is_foreground ? "_in" : "_back") : "";
	const auto prefix_str = QString::fromStdString(info.prefix);
	const auto naming_convention = info.kind == virtual_camera_settings::camera_kind::four_patterns ? ":/test_images/PSI/%1/test_%2%3.tif" : ":/test_images/PSI/%1/f0_t0_i0_ch0_c0_r0_z0_m%2%3.tif";
	auto path = QString(naming_convention).arg(prefix_str).arg(pattern).arg(post_fix);
	if (!skip_read)
	{
		std::cout << "Reading: " << path.toStdString() << std::endl;
		const auto frame = read_sixteen_bit_resource(path, skip_read);
		return frame;
	}
	else
	{
		return tiff_image<unsigned short>();
	}
};

gpu_loaded_frame_set load_simulated_patterns(const virtual_camera_settings& info, bool skip_read = false)
{
	gpu_loaded_frame_set output;
	time_slice ts("Loading Simulated Camera", skip_read);
	const auto is_dpm = info.is_dpm();
	const auto patterns = info.pattern_count();
	{
		for (auto pattern = 0; pattern < patterns; ++pattern)
		{

			const auto img_tiff = read_frame(info, pattern, true, skip_read);
			const auto bg_tiff = is_dpm ? read_frame(info, pattern, false, skip_read) : tiff_image<unsigned short>();
			if (skip_read)
			{
				continue;
			}
			const auto img = load_tiff_img(img_tiff);
			const auto background = load_tiff_img(bg_tiff);
			output.frames.push_back({ img,background });
			static_cast<image_info&>(output) = img_tiff;
		}
	}
	return output;
}

bool virtual_camera_settings::demo_images_support_resize() const noexcept
{
	return !is_dpm() && no_bayer_modes();
}

void virtual_camera_settings::bool_verify_resource_path() const
{
	const auto skip_read = true;
	[[maybe_unused]] auto patterns = load_simulated_patterns(*this, skip_read);
}

void resize_and_load(std::vector<unsigned short>& output, const thrust::device_vector<unsigned short> prototype, image_info in, frame_size out, thrust::device_vector<unsigned short>& temp_array)
{
	if (prototype.empty())
	{
		return;
	}
	NppiSize o_src_size = { in.width,in.height };
	const auto p_src = thrust::raw_pointer_cast(prototype.data());
	const auto n_src_step = o_src_size.width * in.samples_per_pixel * sizeof(unsigned short);
	auto final_width = out.width;
	auto final_height = out.height;
	const auto n_dst_step = final_width * in.samples_per_pixel * sizeof(unsigned short);
	auto p_dst = thrust_safe_get_pointer(temp_array, final_width * final_height * in.samples_per_pixel);
	NppiRect o_src_roi = { 0,0,o_src_size.width,o_src_size.height };
	const NppiRect o_dst_roi = { 0,0,final_width,final_height };
	auto factor_x = final_width / (1.0 * o_src_size.width);
	auto factor_y = final_height / (1.0 * o_src_size.height);
	auto czech_size = [&]
	{
		NppiRect dstrect;
		NPP_SAFE_CALL(nppiGetResizeRect(o_src_roi, &dstrect, factor_x, factor_y, 0, 0, NPPI_INTER_LANCZOS));
		const auto valid_size = !(dstrect.height != final_height || dstrect.width != final_width);
		return valid_size;
	};
	const auto valid_size = czech_size();
	if (!valid_size)
	{
		//zomfg we do a binary search to find a scale factor that nppi won't fuck up munge!
		const auto binary_search = [&](auto grabber_rect, auto grabber_size, auto target_value) {
			auto start = target_value / (1.0 * (grabber_size(o_src_size) + 1));
			auto stop = target_value / (1.0 * (grabber_size(o_src_size) - 1));
			const auto hopes = 10;
			const auto inc = (stop - start) / (1.0 * hopes);
			auto mid_point = 0.0;
			for (auto hope = 0; hope < hopes; ++hope)
			{
				mid_point = (start + stop) / 2;
				auto mid_value = grabber_rect(mid_point);
				if (mid_value == target_value)
				{
					break;
				}
				else if (mid_value > target_value)
				{
					stop = mid_point - inc;
				}
				else
				{
					start = mid_point + inc;
				}
			}
			return mid_point;
		};

		const auto grabber_x_rect = [&](const auto  factor_x_test)
		{
			NppiRect dstrect;
			NPP_SAFE_CALL(nppiGetResizeRect(o_src_roi, &dstrect, factor_x_test, factor_y, 0, 0, NPPI_INTER_LANCZOS));
			return dstrect.width;
		};
		const auto grabber_x_size = [](const NppiSize& size) {return size.width; };
		const auto grabber_y_rect = [&](const auto  factor_y_test)
		{
			NppiRect dstrect;
			NPP_SAFE_CALL(nppiGetResizeRect(o_src_roi, &dstrect, factor_x, factor_y_test, 0, 0, NPPI_INTER_LANCZOS));
			return dstrect.height;
		};
		const auto grabber_y_size = [](const NppiSize& size) {return size.height; };
		factor_x = binary_search(grabber_x_rect, grabber_x_size, final_width);
		factor_y = binary_search(grabber_y_rect, grabber_y_size, final_height);
		if (!czech_size())
		{
			qli_runtime_error("Rounding error during NPP resizing");
		}
	}
	//OKAY FUCKING GREAT NOW WE HAVE THE RIGHT SIZE
	{
		const auto resize_function = (in.samples_per_pixel == 3) ? nppiResizeSqrPixel_16u_C3R : nppiResizeSqrPixel_16u_C1R;
		NPP_SAFE_CALL(resize_function(p_src, o_src_size, n_src_step, o_src_roi, p_dst, n_dst_step, o_dst_roi, factor_x, factor_y, 0, 0, NPPI_INTER_LANCZOS));
		const auto elements = temp_array.size();
		output.resize(elements);
		CUDASAFECALL(cudaMemcpy(output.data(), p_dst, elements * sizeof(unsigned short), cudaMemcpyDeviceToHost));
	}
}

gpu_loaded_frame_set virtual_camera_device::make_aois(const virtual_camera_type& camera_type)
{
	const auto settings = virtual_camera_settings::settings.at(camera_type);
	const auto patterns = load_simulated_patterns(settings);
	//generate them all first
	const auto supports_resize = settings.demo_images_support_resize();
	if (supports_resize)
	{
		for (auto bin_size : { 1,2 })
		{
			bin_modes.push_back(camera_bin(bin_size));
		}
	}
	else
	{
		bin_modes.push_back(camera_bin(1));
	}
	static_cast<camera_contrast_features&>(*this) = settings;
	const image_info prototype_size = patterns;
	if (supports_resize)
	{
		for (auto sampling : { 1,2,4 })
		{
			auto round_to_sixteen = [](auto input)
			{
				return 16 * floor(input / 16);
			};
			
			const auto new_width = round_to_sixteen(prototype_size.width / sampling);
			const auto new_height = round_to_sixteen(prototype_size.height / sampling);
			auto aoi = camera_aoi(new_width, new_height);
			aoi.re_center_and_fixup(prototype_size.width, prototype_size.height, 4);
			aois.push_back(aoi);
		}
	}
	else
	{
		aois.push_back(camera_aoi(prototype_size.width, prototype_size.height));
	}
	if (aois.empty())
	{
		qli_runtime_error();
	}
	static_cast<camera_contrast_features&>(*this) = settings;
	return patterns;
}

void virtual_camera_device::make_virtual_images(const gpu_loaded_frame_set& patterns)
{
	pattern_count = patterns.frames.size();
	const auto total_prepared_patterns = pattern_count * bin_modes.size() * aois.size();
	prepared_images_.resize(total_prepared_patterns);
	thrust::device_vector<unsigned short> temp;
	//
	for (auto bin_idx = 0; bin_idx < bin_modes.size(); ++bin_idx)
	{
		const auto bin = bin_modes.at(bin_idx).s;
		for (auto aoi_idx = 0; aoi_idx < aois.size(); ++aoi_idx)
		{
			const auto aoi_size = aois.at(aoi_idx).to_frame_size();
			const auto size = frame_size(aoi_size.width / bin, aoi_size.height / bin);
			const auto info = image_info(size, patterns.samples_per_pixel, image_info::complex::no);
			for (auto pattern_idx = 0; pattern_idx < pattern_count; ++pattern_idx)
			{
				const auto empty_check = [](virtual_camera_image& img) ->virtual_camera_image&
				{
					if (!img.img.empty())
					{
						qli_runtime_error("Bug in image loading loop");
					}
					return img;
				};
				const auto& prototype = patterns.frames.at(pattern_idx);
				auto& img_buffer = empty_check(get_prepared_image(pattern_idx, aoi_idx, bin_idx, false));
				resize_and_load(img_buffer.img, prototype.img, patterns, info, temp);
				static_cast<image_info&>(img_buffer) = info;
				auto& bg_buffer = empty_check(get_prepared_image(pattern_idx, aoi_idx, bin_idx, true));
				resize_and_load(bg_buffer.img, prototype.background, patterns, info, temp);
				static_cast<image_info&>(bg_buffer) = info;
				if (img_buffer != size)
				{
					qli_runtime_error("Some kind of resizing error");
				}
			}
		}
	}
	//
	blank_frame = virtual_camera_image(patterns);
}

template<class T>
constexpr const T& clampy( const T& v, const T& lo, const T& hi )
{
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

virtual_camera_image& virtual_camera_device::get_prepared_image(int pattern, int aoi, int bin, bool background)
{
	const auto insert_blank_frame  = ((pattern < 0 ) || (pattern >=pattern_count)) ;
	pattern = clampy(pattern,0,pattern_count-1);
	const auto idx = bin + bin_modes.size() * aoi + (bin_modes.size() * aois.size()) * pattern;
	if (bin >= bin_modes.size() || aoi >= aois.size() || idx >= prepared_images_.size())
	{
		qli_runtime_error();
	}
	auto& item = prepared_images_.at(idx);
	if (insert_blank_frame)
	{
		blank_frame.resize(item.img);
		return blank_frame;
	}
	return background ? item.bg : item.img;
}