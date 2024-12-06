#include "compute_engine_demosaic.h"
#include "thrust_resize.h"
#include "cuda_error_check.h"
#include "write_debug_gpu.h"
#include "write_tif.h"
#include <boost/container/static_vector.hpp>

__global__ void FuzzKernel(unsigned short* fuzz_me, const int n_total, const unsigned int seed_xor, const unsigned short constant_offset)
{
	// Get our global thread ID
	// Make sure we do not go out of bounds
	auto id_start = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int seed = id_start;
	const auto some_code_shamelessly_stolen_for_so_or_maybe_google = 0x27d4eb2d;
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= (some_code_shamelessly_stolen_for_so_or_maybe_google ^ seed_xor);
	seed = seed ^ (seed >> 15);
	//pragma funroll
	for (auto offset = 0; offset < 4; ++offset)
	{
		auto idx = id_start + offset;
		if (idx < n_total)
		{
			const unsigned char byte_of_noise = (seed >> 24) & 0xFF;
			const unsigned int value = fuzz_me[idx];
			//fuzz_me[idx] = umax(value + (constant_offset + byte_of_noise), 65535);
			fuzz_me[idx] = umin(constant_offset + (value + +byte_of_noise), 65535);
		}
	}
}

__global__ void rgb_demosaic(cudaTextureObject_t tex, unsigned short* output_buffer, const int rows, const int cols)
{
	const int col = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	const int row = 2 * ((blockIdx.y * blockDim.y) + threadIdx.y);
	if (col < cols && row < rows)
	{
		const auto get = [tex](auto here_row, auto here_col, auto offset_row, auto offset_col)
		{
			const auto point = tex2D<unsigned short>(tex, here_col + offset_col, here_row + offset_row);
			return static_cast<float>(point);
		};
		const auto out = [&](auto row, auto col, auto channel)
		{
			const auto samples_per_pixel = 3;
			const auto idx = row * cols * samples_per_pixel + col * samples_per_pixel + channel;
			return idx;
		};

		const auto I =
			(-1) * get(row + 0, col + 0, -2, +0)
			+ (+2) * get(row + 0, col + 0, -1, +0)
			+ (-1) * get(row + 0, col + 0, +0, -2) + (+2) * get(row + 0, col + 0, +0, -1) + (+4) * get(row + 0, col + 0, +0, +0) + (+2) * get(row + 0, col + 0, +0, +1) + (-1) * get(row + 0, col + 0, +0, +2)
			+ (+2) * get(row + 0, col + 0, +1, +0)
			+ (-1) * get(row + 0, col + 0, +2, +0);

		const auto II =
			(-1) * get(row + 1, col + 1, -2, +0)
			+ (+2) * get(row + 1, col + 1, -1, +0)
			+ (-1) * get(row + 1, col + 1, +0, -2) + (+2) * get(row + 1, col + 1, +0, -1) + (+4) * get(row + 1, col + 1, +0, +0) + (+2) * get(row + 1, col + 1, +0, +1) + (-1) * get(row + 1, col + 1, +0, +2)
			+ (+2) * get(row + 1, col + 1, +1, +0)
			+ (-1) * get(row + 1, col + 1, +2, +0);

		const auto III =
			(+0.5f) * get(row + 0, col + 1, -2, +0)
			+ (-1) * get(row + 0, col + 1, -1, -1) + (-1) * get(row + 0, col + 1, -1, +1)
			+ (-1) * get(row + 0, col + 1, +0, -2) + (4) * get(row + 0, col + 1, +0, -1) + (5) * get(row + 0, col + 1, +0, +0) + (4) * get(row + 0, col + 1, +0, +1) + (-1) * get(row + 0, col + 1, +0, +2)
			+ (-1) * get(row + 0, col + 1, +1, -1) + (-1) * get(row + 0, col + 1, +1, +1)
			+ (+0.5f) * get(row + 0, col + 1, +2, +0);

		const auto IV =
			(-1) * get(row + 1, col + 0, -2, +0)
			+ (-1) * get(row + 1, col + 0, -1, -1) + (4) * get(row + 1, col + 0, -1, +0) + (-1) * get(row + 1, col + 0, -1, +1)
			+ (0.5f) * get(row + 1, col + 0, +0, -2) + (5) * get(row + 1, col + 0, +0, +0) + (0.5f) * get(row + 1, col + 0, +0, +2)
			+ (-1) * get(row + 1, col + 0, +1, -1) + (4) * get(row + 1, col + 0, +1, +0) + (-1) * get(row + 1, col + 0, +1, +1)
			+ (-1) * get(row + 1, col + 0, +2, +0);
		const auto V =
			(-1.5f) * get(row + 1, col + 1, -2, +0)
			+ (2) * get(row + 1, col + 1, -1, -1) + (2) * get(row + 1, col + 1, -1, +1)
			+ (-1.5f) * get(row + 1, col + 1, +0, -2) + (6) * get(row + 1, col + 1, +0, +0) + (-1.5f) * get(row + 1, col + 1, +0, +2)
			+ (2) * get(row + 1, col + 1, +1, -1) + (2) * get(row + 1, col + 1, +1, +1)
			+ (-1.5f) * get(row + 1, col + 1, +2, +0);

		const auto VI =
			(+0.5f) * get(row + 1, col + 0, -2, +0)
			+ (-1) * get(row + 1, col + 0, -1, -1) + (-1) * get(row + 1, col + 0, -1, +1)
			+ (-1) * get(row + 1, col + 0, 0, -2) + (4) * get(row + 1, col + 0, +0, -1) + (5) * get(row + 1, col + 0, +0, +0) + (4) * get(row + 1, col + 0, +0, +1) + (-1) * get(row + 1, col + 0, +0, +2)
			+ (-1) * get(row + 1, col + 0, +1, -1) + (-1) * get(row + 1, col + 0, +1, +1)
			+ (+0.5f) * get(row + 1, col + 0, +2, +0);

		const auto VII =
			(-1) * get(row + 0, col + 1, -2, +0)
			+ (-1) * get(row + 0, col + 1, -1, -1) + (4) * get(row + 0, col + 1, -1, +0) + (-1) * get(row + 0, col + 1, -1, +1)
			+ (0.5f) * get(row + 0, col + 1, +0, -2) + (5) * get(row + 0, col + 1, +0, +0) + (0.5f) * get(row + 0, col + 1, +0, +2)
			+ (-1) * get(row + 0, col + 1, +1, -1) + (4) * get(row + 0, col + 1, +1, +0) + (-1) * get(row + 0, col + 1, +1, +1)
			+ (-1) * get(row + 0, col + 1, +2, +0);

		const auto VIII =
			(-3 / 2.f) * get(row + 0, col + 0, -2, +0)
			+ (2) * get(row + 0, col + 0, -1, -1) + (2) * get(row + 0, col + 0, -1, +1)
			+ (-1.5f) * get(row + 0, col + 0, +0, -2) + (6) * get(row + 0, col + 0, +0, +0) + (-1.5f) * get(row + 0, col + 0, +0, +2)
			+ (2) * get(row + 0, col + 0, +1, -1) + (2) * get(row + 0, col + 0, +1, +1)
			+ (-3 / 2.f) * get(row + 0, col + 0, +2, +0);
#define fix_up(val) (val/8)
		//RGGB
		output_buffer[out(row + 0, col + 0, 0)] = roundf(get(row, col, +0, +0));
		output_buffer[out(row + 0, col + 0, 1)] = roundf(fix_up(I));
		output_buffer[out(row + 0, col + 0, 2)] = roundf(fix_up(VIII));

		output_buffer[out(row + 0, col + 1, 0)] = roundf(fix_up(III));
		output_buffer[out(row + 0, col + 1, 1)] = roundf(get(row, col, +0, +1));
		output_buffer[out(row + 0, col + 1, 2)] = roundf(fix_up(VII));

		output_buffer[out(row + 1, col + 0, 0)] = roundf(fix_up(IV));
		output_buffer[out(row + 1, col + 0, 1)] = roundf(get(row, col, +1, +0));
		output_buffer[out(row + 1, col + 0, 2)] = roundf(fix_up(VI));

		output_buffer[out(row + 1, col + 1, 0)] = roundf(fix_up(V));
		output_buffer[out(row + 1, col + 1, 1)] = roundf(fix_up(II));
		output_buffer[out(row + 1, col + 1, 2)] = roundf(get(row, col, +1, +1));
	}
}

frame_size demosaic_structs::demosaic_rggb_14(input_buffer& output_buffer, const camera_frame<unsigned short>& input_image_h)
{
	const auto size_changed = demosaic_last_input_ != input_image_h;
	if (size_changed)
	{
		demosaic_buffer_.allocate(input_image_h);
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		std::fill(texDesc.addressMode, texDesc.addressMode + 3, cudaAddressModeMirror);
		CUDASAFECALL(cudaCreateTextureObject(&demosaic_buffer_tex_, &demosaic_buffer_, &texDesc, NULL));
		demosaic_last_input_ = input_image_h;
	}
	const auto block_size = 8;
	dim3 block(block_size, block_size, 1);
	auto const divide = [](auto numerator, auto denominator) {return static_cast<unsigned int>(ceil(numerator / (denominator * 1.0f))); };
	dim3 grid(divide(input_image_h.width / 2, block.x), divide(input_image_h.height / 2, block.y), 1);
	CUDASAFECALL(cudaMemcpy2D(demosaic_buffer_.res.pitch2D.devPtr, demosaic_buffer_.res.pitch2D.pitchInBytes, input_image_h.img, input_image_h.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.height, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_DEBUG_SYNC();
	const auto elements = demosaic_last_input_.n() * 3;
	auto output_ptr = thrust_safe_get_pointer(output_buffer, elements);
	rgb_demosaic << <grid, block >> > (demosaic_buffer_tex_, output_ptr, input_image_h.height, input_image_h.width);
	return input_image_h;
}

void demosaic_structs::fuzz(const demosaic_info& patterns)
{
#if _DEBUG
	for (auto idx : patterns.frames_made)
	{
		auto& input_frame = idx < input_frames.size() ? input_frames.at(idx) : input_frames.front();
		{
			static int gridSize, blockSize;
			static auto old_size = (-1);
			const auto size_changed = (input_frame.size() != old_size);
			if (size_changed)
			{
				int minGridSize;//unused?
				CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, FuzzKernel, 0, 0));//todo bug here on the type!!!
				gridSize = ((ceil(input_frame.size() / 4)) + blockSize - 1) / blockSize;
				old_size = input_frame.size();
			}
			// ReSharper disable once CppLocalVariableMayBeConst
			auto ptr = thrust::raw_pointer_cast(input_frame.data());
			const auto time_since_epoch = std::chrono::system_clock::now().time_since_epoch();
			const auto micro_seconds_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(time_since_epoch).count();
			const auto mili_seconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count();
			const unsigned short seconds_since_epoch = 2000 * (1 + sin(mili_seconds_since_epoch / 150));
			FuzzKernel << < gridSize, blockSize >> > (ptr, old_size, time_since_epoch.count(), 0);
			CUDA_DEBUG_SYNC();
		}
	}
#endif
}

int demosaic_structs::pattern_to_load(const phase_retrieval retrieval, const int pattern_idx)
{
	const auto is_special = phase_retrieval_setting::settings.at(retrieval).modulator_patterns() == pattern_count_from_file;
	return  is_special ? 0 : pattern_idx;
}

demosaic_info demosaic_structs::load_resize_and_demosaic(const camera_frame<unsigned short>& input_image_h, const processing_quad& quad, bool is_live)
{
	//
	auto demosaic_mode = quad.demosaic;
	auto retrieval = quad.retrieval;
	auto show_only_raw_first_frame = is_live && (quad.processing == phase_processing::raw_frames);
	const auto apply_polarization_renumbering = phase_retrieval_setting::settings.at(retrieval).special_polarizer_paths && (!show_only_raw_first_frame);
	const auto skip_demosaicing = demosaic_mode == demosaic_mode::no_processing;
	const auto load_this_pattern = pattern_to_load(retrieval, input_image_h.pattern_idx);
	const auto black_and_white = 1, color = 3;
	auto pol_pass_through_fall_back = [&](polarizer_demosaic_kind kind)
	{
		auto& output_buffer = input_frames.at(load_this_pattern);
		auto frame_size = demosaic_polarizer_pass_one(output_buffer, input_image_h, kind);
		return demosaic_info({ input_image_h.pattern_idx }, frame_size, black_and_white);
	};
	if (skip_demosaicing)
	{
		auto& output_buffer = input_frames.at(load_this_pattern);
		const auto elements = input_image_h.n() * input_image_h.samples_per_pixel;
		thrust_safe_resize(output_buffer, elements);
		thrust::copy(input_image_h.img, input_image_h.img + elements, output_buffer.begin());
		return demosaic_info({ input_image_h.pattern_idx }, input_image_h, input_image_h.samples_per_pixel);
	}
	else if (demosaic_mode == demosaic_mode::rggb_14_native)
	{
		auto& output_buffer = input_frames.at(load_this_pattern);
		auto frame_size = demosaic_rggb_14(output_buffer, input_image_h);
		return demosaic_info({ input_image_h.pattern_idx }, frame_size, color);
	}
	else if (demosaic_mode == demosaic_mode::polarization_0_45_90_135)
	{
		// if you do it on a polarization mode you get out 4 images, if you do it on a regular mode you get out regular images
		if (apply_polarization_renumbering)
		{
			auto offset = input_image_h.pattern_idx * 4;
			demosaic_info::frames  normal = { offset + 0, offset + 1, offset + 2, offset + 3 };
			auto frame_size = demosaic_polarizer_0_45_90_135(input_frames.at(normal.at(0)), input_frames.at(normal.at(1)), input_frames.at(normal.at(2)), input_frames.at(normal.at(3)), input_image_h);
			return demosaic_info(normal, frame_size, black_and_white);
		}
		else
		{
			return pol_pass_through_fall_back(polarizer_demosaic_kind::p90);
		}
	}
	else if ((demosaic_mode == demosaic_mode::polarization_0_90) || (demosaic_mode == demosaic_mode::polarization_45_135))
	{
		const auto is_a_polarization_mode = phase_retrieval_setting::settings.at(retrieval).special_polarizer_paths;
		if (apply_polarization_renumbering)
		{
			auto offset = input_image_h.pattern_idx * 2;
			boost::container::static_vector<int, 4>  normal = { offset + 0, offset + 1 };
			auto frame_size = demosaic_polarizer_doubles(input_frames.at(normal.at(0)), input_frames.at(normal.at(1)), input_image_h, demosaic_mode);
			return demosaic_info(normal, frame_size, black_and_white);
		}
		else
		{
			auto hack = (demosaic_mode == demosaic_mode::polarization_0_90) ? polarizer_demosaic_kind::p90 : polarizer_demosaic_kind::p45;
			return pol_pass_through_fall_back(hack);
		}
	}
	qli_not_implemented();
}

int processing_quad::frames_per_demosaic(bool is_live) const
{
	//FYI this is pretty fucked, simplify later
	//Polarization
	// 1 frame to [2,4], if path exists
	//else 1 frame
	auto show_only_raw_first_frame = is_live && (processing == phase_processing::raw_frames);
	const auto apply_polarization_renumbering = phase_retrieval_setting::settings.at(retrieval).special_polarizer_paths && (!show_only_raw_first_frame);
	const auto skip_demosaicing = demosaic == demosaic_mode::no_processing;
	auto pol_pass_through_fall_back = [&](polarizer_demosaic_kind kind)
	{
		return 1;
	};
	if (skip_demosaicing)
	{
		return 1;
	}
	else if (demosaic == demosaic_mode::rggb_14_native)
	{
		return 1;
	}
	else if (demosaic == demosaic_mode::polarization_0_45_90_135)
	{
		// if you do it on a polarization mode you get out 4 images, if you do it on a regular mode you get out regular images
		if (apply_polarization_renumbering)
		{
			return 4;
		}
		else
		{
			return pol_pass_through_fall_back(polarizer_demosaic_kind::p90);
		}
	}
	else if ((demosaic == demosaic_mode::polarization_0_90) || (demosaic == demosaic_mode::polarization_45_135))
	{
		const auto is_a_polarization_mode = phase_retrieval_setting::settings.at(retrieval).special_polarizer_paths;
		if (apply_polarization_renumbering)
		{
			return 2;
		}
		else
		{
			return 1;
		}
	}
	qli_runtime_error("Should not happen");
}