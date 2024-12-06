#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include "ml_shared.h"
#include "ml_transformer.h"
#include "program_config.h"
#if INCLUDE_ML==1
#include <npp_non_owning_buffer.h>
#include "device_factory.h"
#include <npp.h>
#include "npp_error_check.h"
#include "thrust_resize.h"
#include "approx_equals.h"
#include "write_debug_gpu.h"
#include <NvInfer.h>
#include "unet_engine_v1.h"
#include "unet_engine_v2.h"
#include "unet_engine_v3.h"
#include "ueffunet_engine_onnx_v1.h"
#include <boost/noncopyable.hpp>
#include "clamp_and_scale.h"
#include "scale.h"
#include <algorithm>
#include "ml_timing.h"
#include <itaSettings.h>
#pragma comment(lib, "nvinfer.lib")

using std::cout;
using std::endl;

//class Logger : public nvinfer1::ILogger
//{
//	void log(Severity severity, const char* msg) override
//	{
//#if _DEBUG
//		// suppress info-level messages
//		if (severity != Severity::kINFO)
//			std::cout << msg << std::endl;
//#endif
//	}
//} gLogger;

struct ml_engine : private boost::noncopyable
{
	virtual void do_inference(float* output, float* input) = 0;
	virtual ~ml_engine() = default;
};

//class Logger : public nvinfer1::ILogger
//{
//	void log(Severity severity, const char* msg) override
//	{
//#if _DEBUG
//		// suppress info-level messages
//		if (severity != Severity::kINFO)
//			std::cout << msg << std::endl;
//#endif
//	}
//} gLogger;

Logger gLogger;

template<typename T>
__global__ void _four_channel_demux(T* output, const T* four_channel, const int numel)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		auto v1 = four_channel[numel * 0 + idx];
		auto v2 = four_channel[numel * 1 + idx];
		auto v3 = four_channel[numel * 2 + idx];
		auto v4 = four_channel[numel * 3 + idx];
		auto max1 = thrust::max(v1, v2);
		auto max2 = thrust::max(v3, v4);
		auto max = thrust::max(max1, max2);
		if (max == v1)
		{
			output[idx] = 0;
		}
		else if (max == v2)
		{
			output[idx] = 1 * 85;
		}
		else if (max == v3)
		{
			output[idx] = 2 * 85;
		}
		else
		{
			output[idx] = 3 * 85;
		}
	}
}

template<typename T>
__global__ void _four_channel_demux_nhwc(T* output, const T* four_channel, const int in_h, const int in_w, const bool shift = false)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	//  Need to make sure this thread corresponds to a valid pixel on the original graph
	if (y < in_h && x < in_w) {
		//  Each thread is responsible for one pixel in all channels, [y][x] goes to [2y][2x]~[2y+1][2x+1]
		auto idx = (y * in_w + x) * 4;
		auto outidx = (y * in_w + x);
		auto v1 = four_channel[idx + 0];
		auto v2 = four_channel[idx + 1];
		auto v3 = four_channel[idx + 2];
		auto v4 = four_channel[idx + 3];
		auto max1 = thrust::max(v1, v2);
		auto max2 = thrust::max(v3, v4);
		auto max = thrust::max(max1, max2);
		if (max == v1)
		{
			output[outidx] = (!shift)?0:(1*85);
		}
		else if (max == v2)
		{
			output[outidx] = (!shift)?(1 * 85):(2*85);
		}
		else if (max == v3)
		{
			output[outidx] = (!shift)?(2 * 85):(3*85);
		}
		else
		{
			output[outidx] = (!shift)?(3 * 85):(0);
		}
	}
}

struct thresholder_functor
{
	//  tell  CUDA that the following code can be executed on the CPU and the GPU
	__host__ __device__  float  operator()(const float& x) const
	{
		const auto threshold = 0.1;
		return x > threshold ? 1 : 0;
	}
};
struct dummy_pass_through_network final : ml_engine
{
	const int numel;
	explicit dummy_pass_through_network(const int numel) : numel(numel)
	{

	}
	void do_inference(float* output, float* input) override
	{
		//const auto count_bytes = numel * sizeof(float);
		//CUDASAFECALL(cudaMemcpy(output, input, count_bytes, cudaMemcpyDeviceToDevice));
		auto output_begin = thrust::device_pointer_cast(output);
		auto input_begin = thrust::device_pointer_cast(input);
		thrust::transform(input_begin, input_begin + numel, output_begin, thresholder_functor());
	}
};

struct qli_tensor_rt_engine : ml_engine
{
	int width, height, channels;

	qli_tensor_rt_engine(const int width, const int height, const int channels) : width(width), height(height), channels(channels), engine(nullptr), builder(nullptr), context(nullptr)
	{
		LOGGER_INFO("width: " << width << ", height: " << height <<  ", channels:" << channels);
	}
	virtual  ~qli_tensor_rt_engine()
	{
		//Two condoms, one for your protection one for her protection
		CUDASAFECALL(cudaDeviceSynchronize());
		context->destroy();
		engine->destroy();
		builder->destroy();
		CUDASAFECALL(cudaDeviceSynchronize());
	}
protected:
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IBuilder* builder;
	nvinfer1::IExecutionContext* context;
	static bool stream_not_initialized;
	//leaked
	static cudaStream_t stream;
};

cudaStream_t qli_tensor_rt_engine::stream;
bool qli_tensor_rt_engine::stream_not_initialized = true;

struct qli_unet_v1_semantic final : qli_tensor_rt_engine
{
	thrust::device_vector<float> internal_buffer;
	void four_channel_demux(float* output, const float* four_channel_muxed, const int numel)
	{
		static int grid_size, block_size;
		static auto old_size = (-1);
		const auto size_changed = (numel != old_size);
		if (size_changed)
		{
			int min_grid_size;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, _four_channel_demux<float>, 0, 0));//todo bug here on the type!!!
			grid_size = (numel + block_size - 1) / block_size;
		}
		_four_channel_demux<float> << <grid_size, block_size >> > (output, four_channel_muxed, numel);
	}
	explicit qli_unet_v1_semantic(const std::string& filename, const int width, const int height, const int channels) : qli_tensor_rt_engine(width, height, channels)
	{
		if (stream_not_initialized)
		{
			CUDASAFECALL(cudaStreamCreate(&stream));
			stream_not_initialized = false;
		}
		builder = nvinfer1::createInferBuilder(gLogger);
		if (!builder)
		{
			qli_runtime_error("Failed to Make Builder");
		}
		engine = create_unet_engine_v1(builder, filename.c_str());
		if (!engine)
		{
			qli_runtime_error("Failed to Make Engine");
		}
		context = engine->createExecutionContext();
	}
	void do_inference(float* output, float* input) override
	{
		const auto batchSize = 1;
		const auto temp_buffer_elements = width * height * channels;
		auto temp_output = thrust_safe_get_pointer(internal_buffer, temp_buffer_elements);
		void* buffers[2] = { input ,temp_output };
		CUDASAFECALL(cudaDeviceSynchronize());
		const auto successful_launch = context->enqueue(batchSize, buffers, stream, nullptr);
		if (!successful_launch)
		{
			qli_runtime_error("Failed to enqueue kernels");
		}
		CUDASAFECALL(cudaStreamSynchronize(stream));
		const auto output_image_size = width * height;
		four_channel_demux(output, temp_output, output_image_size);
	}
};


struct qli_unet_v2_mapping final : qli_tensor_rt_engine
{
	int width, height;
	explicit qli_unet_v2_mapping(const std::string& filename, const int width, const int height, const float x_min, const float x_max) : qli_tensor_rt_engine(width, height, 1)
	{
		if (stream_not_initialized)
		{
			CUDASAFECALL(cudaStreamCreate(&stream));
			stream_not_initialized = false;
		}
		builder = nvinfer1::createInferBuilder(gLogger);
		if (!builder)
		{
			qli_runtime_error("Failed to Make Builder");
		}
		engine = create_unet_engine_v2(builder, filename.c_str(), 1, height, width, x_min, x_max);
		if (!engine)
		{
			qli_runtime_error("Failed to Make Engine");
		}
		context = engine->createExecutionContext();
	}

	void do_inference(float* output, float* input) override
	{
		const auto batchSize = 1;
		void* buffers[2] = { input ,output };
		CUDASAFECALL(cudaDeviceSynchronize());
		const auto successful_launch = context->enqueue(batchSize, buffers, stream, nullptr);
		if (!successful_launch)
		{
			qli_runtime_error("Failed to enqueue kernels");
		}
		CUDASAFECALL(cudaStreamSynchronize(stream));
	}

};


struct qli_unet_v3_mapping final : qli_tensor_rt_engine
{
	int width, height;
	explicit qli_unet_v3_mapping(const std::string& filename, const int width, const int height, const float x_min, const float x_max) : qli_tensor_rt_engine(width, height, 1)
	{
		if (stream_not_initialized)
		{
			CUDASAFECALL(cudaStreamCreate(&stream));
			stream_not_initialized = false;
		}
		builder = nvinfer1::createInferBuilder(gLogger);
		if (!builder)
		{
			qli_runtime_error("Failed to Make Builder");
		}
		engine = create_unet_engine_v3(builder, filename.c_str(), 1, height, width, x_min, x_max);
		if (!engine)
		{
			qli_runtime_error("Failed to Make Engine");
		}
		context = engine->createExecutionContext();
	}

	void do_inference(float* output, float* input) override
	{
		const auto batchSize = 1;
		void* buffers[2] = { input ,output };
		CUDASAFECALL(cudaDeviceSynchronize());
		const auto successful_launch = context->enqueue(batchSize, buffers, stream, nullptr);
		if (!successful_launch)
		{
			qli_runtime_error("Failed to enqueue kernels");
		}
		CUDASAFECALL(cudaStreamSynchronize(stream));
	}

};

struct qli_ueffnet_v1_mapping final : qli_tensor_rt_engine
{
	int width, height;
	explicit qli_ueffnet_v1_mapping(const std::string& filename, const int width, const int height, const float x_min, const float x_max) : qli_tensor_rt_engine(width, height, 1)
	{
		if (stream_not_initialized)
		{
			CUDASAFECALL(cudaStreamCreate(&stream));
			stream_not_initialized = false;
		}
		builder = nvinfer1::createInferBuilder(gLogger);
		if (!builder)
		{
			qli_runtime_error("Failed to Make Builder");
		}
		engine = buildONNXModel(builder, filename.c_str(), gLogger);
		if (!engine)
		{
			qli_runtime_error("Failed to Make Engine");
		}
		context = engine->createExecutionContext();
	}

	void do_inference(float* output, float* input) override
	{
		const auto batchSize = 1;
		void* buffers[2] = { input ,output };
		CUDASAFECALL(cudaDeviceSynchronize());
		const auto start = ml_quick_timestamp();
		const auto successful_launch = context->enqueue(batchSize, buffers, stream, nullptr);
		const auto end = ml_quick_timestamp();
		std::cout << "ML Inference only Took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count() << " ms" << std::endl;
		if (!successful_launch)
		{
			qli_runtime_error("Failed to enqueue kernels");
		}
		CUDASAFECALL(cudaStreamSynchronize(stream));
	}

};

//template<typename T>
//struct npp_non_owning_buffer : npp_dimensions
//{
//	T* buffer;
//	npp_non_owning_buffer(T* buffer, const int nSrcStep, const NppiSize& Size, const NppiRect& ROI) : npp_dimensions(nSrcStep, Size, ROI), buffer(buffer) {}
//	npp_non_owning_buffer(T* buffer, const npp_dimensions& npp_dimensions) : npp_dimensions(npp_dimensions), buffer(buffer) {}
//	npp_non_owning_buffer(T* buffer, const NppiSize& Size) : npp_dimensions(Size.width * sizeof(T), Size, { 0,0,Size.width,Size.height }), buffer(buffer) {}
//	npp_non_owning_buffer(T* buffer, const frame_size& Size) : npp_non_owning_buffer(buffer, NppiSize{ Size.width,Size.height }) {}
//	static npp_non_owning_buffer safe_from_buffer(thrust::device_vector<T>& buffer_to_use, const frame_size& dimensions)
//	{
//		const NppiSize dimensions_npp = { dimensions.width,dimensions.height };
//		return safe_from_buffer(buffer_to_use, dimensions_npp);
//	}
//	static npp_non_owning_buffer safe_from_buffer(thrust::device_vector<T>& buffer_to_use, const npp_dimensions& dimensions)
//	{
//		auto right_size = safe_from_buffer(buffer_to_use, dimensions.Size);
//		static_cast<npp_dimensions&>(right_size) = dimensions;
//		return right_size;
//	}
//	static npp_non_owning_buffer safe_from_buffer(thrust::device_vector<T>& buffer_to_use, const NppiSize& size)
//	{
//		const auto number_of_elements = size.width * size.height;
//		const auto input_ptr = thrust_safe_get_pointer(buffer_to_use, number_of_elements);
//		return npp_non_owning_buffer(input_ptr, size);
//	}
//	void write_full(const std::string& name, bool do_write) const
//	{
//		CUDASAFECALL(cudaDeviceSynchronize());
//		auto start = buffer;
//		const auto pitch_numel = nStep / sizeof(T);
//		write_debug_gpu_with_pitch(start, Size.width, Size.height, pitch_numel, 1, name.c_str(), do_write);
//	}
//	void write(const std::string& name, bool do_write) const
//	{
//		CUDASAFECALL(cudaDeviceSynchronize());
//		auto start = buffer + Size.width * ROI.y + ROI.x;
//		const unsigned long long pitch_numel = nStep / sizeof(T);
//		write_debug_gpu_with_pitch(start, ROI.width, ROI.height, pitch_numel, 1, name.c_str(), do_write);
//	}
//	auto thrust_begin()
//	{
//		return thrust::device_pointer_cast(buffer);
//	}
//	auto thrust_end()
//	{
//		const auto numel = Size.width * Size.height;
//		return thrust::device_pointer_cast(buffer + numel);
//	}
//};

template<class T>
constexpr const T& not_std_clamp(const T& v, const T& lo, const T& hi)
{
	//because clamp is missing
	return (v < lo) ? lo : (hi < v) ? hi : v;
}


frame_size ml_transformer::get_ml_output_size(const frame_size& camera_frame, const float input_pixel_ratio, const ml_remapper_file::ml_remapper_types ml_type, const bool skip_scale)
{
	if (ml_type == ml_remapper_file::ml_remapper_types::off || skip_scale)
	{
		return camera_frame;
	}
	const auto& ml_settings_file = ml_remapper_file::ml_remappers.at(ml_type);
	auto scale_ratio = input_pixel_ratio / ml_settings_file.designed_pixel_ratio;
//#if _DEBUG
//	scale_ratio = 1.0;
//#endif
	const auto network_size = ml_settings_file.get_network_size();
	const auto max_image = frame_size(round_up_division(network_size.width, scale_ratio), round_up_division(network_size.height, scale_ratio));
	auto actual_image = frame_size(round_up_division(camera_frame.width, scale_ratio), round_up_division(camera_frame.height, scale_ratio));
	actual_image.width = not_std_clamp(actual_image.width, 0, max_image.width);
	actual_image.height = not_std_clamp(actual_image.height, 0, max_image.height);
#if _DEBUG
	if (actual_image.n() == 0)
	{
		qli_runtime_error("Oh nope.. time to fuck gabi in the ass.");
	}
#endif
	return actual_image;
}

bool ml_transformer::fuck_this_shit(float* destination_ml_pointer, const frame_size& destination_frame_size, const  float* input_ptr, const frame_size& input_frame_size, const ml_remapper& settings, float input_pixel_ratio, const bool skip_ml) {
	LOGGER_INFO("input_frame_size(w, h): " << input_frame_size.width << ", " << input_frame_size.height);


}


bool ml_transformer::do_ml_transform(float* destination_ml_pointer, const frame_size& destination_frame_size, const  float* input_ptr, const frame_size& input_frame_size, const ml_remapper& settings, float input_pixel_ratio, const bool skip_ml)
{
	LOGGER_INFO("input_frame_size(w, h): " << input_frame_size.width << ", " << input_frame_size.height);

	const auto compare_size = [](const NppiSize& A, const NppiSize& B)
	{
		return A.width == B.width && A.height == B.height;
	};
	const auto debug = false;
	LOGGER_INFO("settings.ml_remapper_type: " << (int)settings.ml_remapper_type);
	const auto& ml_file_settings = ml_remapper_file::ml_remappers.at(settings.ml_remapper_type);

	LOGGER_INFO("input_pixel_ratio: " << input_pixel_ratio);
	const bool skip_resample = approx_equals(input_pixel_ratio, ml_file_settings.designed_pixel_ratio);
	LOGGER_INFO("skip_resample: " << skip_resample);
	LOGGER_INFO("input pixel ratio is " << input_pixel_ratio);
	const auto interpolation_mode = NPPI_INTER_CUBIC;

	//resize to output
	//Step 1: resize to match expected pixel ratio size
	const auto resized_img = [&]()
	{
		{
			cudaPointerAttributes attributes;
			cudaPointerGetAttributes(&attributes, input_ptr);
			LOGGER_INFO("input_ptr is pointing to: " << (int)attributes.type);
		}
		const NppiSize input_size = { input_frame_size.width, input_frame_size.height };
		const npp_non_owning_buffer<float> input(const_cast<float*>(input_ptr), input_size);
		input.write("ML_Step_0_Input.tif", debug);
		if (skip_resample)
		{
			return input;
		}
		//Network designed for 40x (7.14), input is 20x (3.57) -> resize by a factor of 2
		LOGGER_INFO("ml_file_settings.designed_pixel_ratio: " << ml_file_settings.designed_pixel_ratio);
		const auto scale_ratio = ml_file_settings.designed_pixel_ratio / input_pixel_ratio;
		LOGGER_INFO("scale_ratio: " << scale_ratio);
		const NppiSize output_size = { round_up_division(input_size.width,scale_ratio),round_up_division(input_size.height,scale_ratio) };
		LOGGER_INFO("output_size(w, h): " << output_size.width << ", " << output_size.height);
		LOGGER_INFO("input.ROI(w, h, x, y): " << input.ROI.width << ", " << input.ROI.height << ", " << input.ROI.x << ", " << input.ROI.y);
		const auto dst = npp_non_owning_buffer<float>::safe_from_buffer(resize_buffer, output_size);
		NPP_SAFE_CALL(nppiResize_32f_C1R(input.buffer, input.nStep, input.Size, input.ROI, dst.buffer, dst.nStep, dst.Size, dst.ROI, interpolation_mode));
		dst.write("ML_Step_1_Resized.tif", debug);
		return dst;
	}();//resize_buffer
	//Step 2: Copy + crop + pad + into ml buffer
	const auto network_size = ml_file_settings.get_network_size();
	LOGGER_INFO("network_size(w, h): " << network_size.width << ", " << network_size.height);
	auto cropped_input_image = [&]() {
		const NppiSize ml_network_input_size = { network_size.width, network_size.height };
		const bool same_size = compare_size(resized_img.Size, ml_network_input_size);
		LOGGER_INFO("resized_img(w, h): " << resized_img.Size.width << ", " << resized_img.Size.height);
		if (same_size)
		{
			return resized_img;
		}
		auto dst = npp_non_owning_buffer<float>::safe_from_buffer(ml_input_buffer, ml_network_input_size);
		//center the ROI
		const auto left_most = [&](const int length, const int max_length)
		{
			const auto value = static_cast<int>(std::floor((max_length - length) / 2));
			const auto value2 = not_std_clamp(value, 0, max_length);
			return value2;
		};
		dst.ROI = {
			left_most(resized_img.Size.width,dst.Size.width),
			left_most(resized_img.Size.height,dst.Size.height),
			not_std_clamp(resized_img.Size.width,0,dst.Size.width),
			not_std_clamp(resized_img.Size.height,0,dst.Size.height)
		};
		//If the ROI is bigger then we're cropping, in which case this is the largest ROI
		const auto top = dst.ROI.y;
		const auto left = dst.ROI.x;
		NPP_SAFE_CALL(nppiCopyWrapBorder_32f_C1R(resized_img.buffer, resized_img.nStep, resized_img.Size, dst.buffer, dst.nStep, dst.Size, top, left));//need to fix this to be the middle
		dst.write("ML_Step_2_1_roi.tif", debug);
		dst.write_full("ML_Step_2_2_padded.tif", debug);
		return dst;
	}();//ml_input_buffer
	//Step 3: Write the phase image at the new size, this required for overlay applications, in the future we'll make this a null copy, or something when no resize happens
	const auto output_size = get_ml_output_size(input_frame_size, input_pixel_ratio, settings.ml_remapper_type, skip_resample);
	if (!skip_resample)
	{
		const auto dst_phase = npp_non_owning_buffer<float>::safe_from_buffer(phase_resized, output_size);
		NPP_SAFE_CALL(nppiResize_32f_C1R(cropped_input_image.buffer, cropped_input_image.nStep, cropped_input_image.Size, cropped_input_image.ROI, dst_phase.buffer, dst_phase.nStep, dst_phase.Size, dst_phase.ROI, interpolation_mode));
		dst_phase.write("ML_Step_3_phase_final_size.tif", debug);
	}
	if (skip_ml)
	{
		return !skip_resample;
	}
	//Step 4: apply scale factors
	if (ml_file_settings.do_input_scale)
	{
		thrust::transform(cropped_input_image.thrust_begin(), cropped_input_image.thrust_end(), cropped_input_image.thrust_begin(), clamp_n_scale<float, float>(ml_file_settings.input_min, ml_file_settings.input_max, 0, 1));
		cropped_input_image.write("ML_Step_4_scaled.tif", debug);
	}
	// todo: insert something here to make the image 3 channel and scaled
	// Added step: do imagenet NCHW preprocessing
	auto ready_input = [&]() {
		if (ml_file_settings.imagenet_preprocessing)
		{
			// copying into three separate buffers for rgb
			auto input_single_size = frame_size{ cropped_input_image.Size.width, cropped_input_image.Size.height };
			auto input_image_r = npp_non_owning_buffer<float>::safe_from_buffer(ml_input_r_buffer, input_single_size);
			auto input_image_g = npp_non_owning_buffer<float>::safe_from_buffer(ml_input_g_buffer, input_single_size);
			auto input_image_b = npp_non_owning_buffer<float>::safe_from_buffer(ml_input_b_buffer, input_single_size);
			NPP_SAFE_CALL(nppiCopy_32f_C1R(cropped_input_image.buffer, cropped_input_image.nStep, input_image_r.buffer, input_image_r.nStep, input_image_r.Size));
			NPP_SAFE_CALL(nppiCopy_32f_C1R(cropped_input_image.buffer, cropped_input_image.nStep, input_image_g.buffer, input_image_g.nStep, input_image_g.Size));
			NPP_SAFE_CALL(nppiCopy_32f_C1R(cropped_input_image.buffer, cropped_input_image.nStep, input_image_b.buffer, input_image_b.nStep, input_image_b.Size));
			input_image_r.write("ML_RGB_test_r.tif", debug);
			input_image_g.write("ML_RGB_test_g.tif", debug);
			input_image_b.write("ML_RGB_test_b.tif", debug);

			// performing normalization on each buffer
			// ImageNet Magic Number
			std::vector<float> means{ 0.485, 0.456, 0.406 };
			std::vector<float> stds{ 0.229, 0.224, 0.225 };
			thrust::transform(input_image_r.thrust_begin(), input_image_r.thrust_end(), input_image_r.thrust_begin(), scale<float, float>(means.at(0), means.at(0) + stds.at(0), 0, 1));
			thrust::transform(input_image_g.thrust_begin(), input_image_g.thrust_end(), input_image_g.thrust_begin(), scale<float, float>(means.at(1), means.at(1) + stds.at(1), 0, 1));
			thrust::transform(input_image_b.thrust_begin(), input_image_b.thrust_end(), input_image_b.thrust_begin(), scale<float, float>(means.at(2), means.at(2) + stds.at(2), 0, 1));

			input_image_r.write("ML_RGB_test_r_norm.tif", debug);
			input_image_g.write("ML_RGB_test_g_norm.tif", debug);
			input_image_b.write("ML_RGB_test_b_norm.tif", debug);


			// padding the data into NCHW (r image --> g image --> b image)
			auto input_rgb_size = frame_size{ cropped_input_image.Size.width, cropped_input_image.Size.height * 3 };
			auto input_image_rgb = npp_non_owning_buffer<float>::safe_from_buffer(ml_input_rgb_buffer, input_rgb_size);
			NPP_SAFE_CALL(nppiCopy_32f_C1R(input_image_r.buffer, input_image_r.nStep, input_image_rgb.buffer, input_image_rgb.nStep, input_image_r.Size));
			NPP_SAFE_CALL(nppiCopy_32f_C1R(input_image_g.buffer, input_image_g.nStep, input_image_rgb.buffer + input_single_size.width * input_single_size.height, input_image_rgb.nStep, input_image_g.Size));
			NPP_SAFE_CALL(nppiCopy_32f_C1R(input_image_b.buffer, input_image_b.nStep, input_image_rgb.buffer + (input_single_size.width * input_single_size.height) * 2, input_image_rgb.nStep, input_image_b.Size));

			input_image_rgb.write("ML_RGB_test_NCHW.tif", debug);
			return input_image_rgb;
		}
		else
		{
			return cropped_input_image;
		}
	}();
	//Step 5: evaluate the ML model
	//auto ml_destination = npp_non_owning_buffer<float>::safe_from_buffer(ml_output_buffer, cropped_input_image);
	auto output_raw_size = frame_size{ cropped_input_image.Size.width, cropped_input_image.Size.height * ml_file_settings.output_channel };
	npp_non_owning_buffer<float> output_raw = npp_non_owning_buffer<float>::safe_from_buffer(ml_output_raw_buffer, output_raw_size);
	{
		auto engine = safe_get_ml_engine(settings.ml_remapper_type);
		engine->do_inference(output_raw.buffer, ready_input.buffer);
		output_raw.write("ML_Step_5_1_NHWC_output.tif", debug); // <- here*******
		//ml_destination.write("ML_Step_5_1_Cropped_Destination.tif", debug);
		//ml_destination.write_full("ML_Step_5_2_Full_Destination.tif", debug);
	}
	// inserted step, need to argmax
	auto ml_destination = [&]() {
		if (ml_file_settings.output_channel == 1)
		{
			return output_raw;
		}
#if FUCKFACEGABRIELPOPESCU
		itaSettings::register_ml_out(std::make_shared<npp_non_owning_buffer<float>>(npp_non_owning_buffer<float>::safe_from_buffer(ml_output_buffer, cropped_input_image)));
		auto& output_ready = *(itaSettings::ml_out);
#else
		auto output_ready = npp_non_owning_buffer<float>::safe_from_buffer(ml_output_buffer, cropped_input_image);
#endif
		// for loop?
		dim3 grid_dim(ceil(cropped_input_image.Size.height / 32.0), ceil(cropped_input_image.Size.width / 32.0));
		dim3 block_dim(32, 32);
		_four_channel_demux_nhwc<float> << <grid_dim, block_dim >> > (output_ready.buffer, output_raw.buffer, cropped_input_image.Size.height, cropped_input_image.Size.width, ml_file_settings.shift_argmax);
		output_ready.write("ML_Step_5_inserted_argmax.tif", debug);
		return output_ready;
	}();
#if FUCKFACEGABRIELPOPESCU
	if (itaSettings::current_trigger_condition != 0) {
		return !skip_resample;
	}
#endif
	//Step 6: rescale back to native scale
	auto up_sampled_ml = [&]
	{
		//grab the ROI
		if (skip_resample)
		{
			return ml_destination;
		} else
		{
			const auto dst_ml = npp_non_owning_buffer<float>::safe_from_buffer(resize_buffer, output_size);
			NPP_SAFE_CALL(nppiResize_32f_C1R(ml_destination.buffer, ml_destination.nStep, ml_destination.Size, ml_destination.ROI, dst_ml.buffer, dst_ml.nStep, dst_ml.Size, dst_ml.ROI, interpolation_mode));
			dst_ml.write("ML_Step_6_Up_Sampled.tif", debug);
			return dst_ml;			
		}
	}();
	//Step 7: Crop into destination this is a the final copy, and the only "unnecessary" deep copy
	//auto final_destination = ml_destination;
	auto final_destination = [&]
	{
		const auto dst = npp_non_owning_buffer<float>(destination_ml_pointer, output_size);
		
		const auto left_most = [](npp_non_owning_buffer<float>& buffer, const int x_left, const int y_left) {
			//some bounds checking here, maybe
			return buffer.buffer + (buffer.nStep / sizeof(float)) * y_left + x_left;
		};
		NPP_SAFE_CALL(nppiCopy_32f_C1R(left_most(up_sampled_ml, up_sampled_ml.ROI.x, up_sampled_ml.ROI.y), up_sampled_ml.nStep, dst.buffer, dst.nStep, dst.Size));
		//auto source = left_most(up_sampled_ml, up_sampled_ml.ROI.x, up_sampled_ml.ROI.y);
		////NPP_SAFE_CALL(nppiCopy_32f_C1R(source, up_sampled_ml.nStep, dst.buffer, dst.nStep, dst.Size));
		//NPP_SAFE_CALL(nppiCopy_32f_C1R(source, up_sampled_ml.nStep, dst.buffer, dst.nStep, up_sampled_ml.Size));
		dst.write("ML_Step_7_Cropped.tif", debug);
		return dst;
	}();
	//Step 8: Rescale ml into output range
	if (ml_file_settings.do_output_scale) {
		thrust::transform(final_destination.thrust_begin(), final_destination.thrust_end(), final_destination.thrust_begin(), clamp_n_scale<float, float>(ml_file_settings.output_min_in, ml_file_settings.output_max_in, ml_file_settings.output_min_out, ml_file_settings.output_max_out));
	}
	final_destination.write("ML_Step_8_Output_Scale.tif", debug);

#if _DEBUG
	if (!compare_size(final_destination.Size, NppiSize({ destination_frame_size.width,destination_frame_size.height })))
	{
		qli_runtime_error("File Size Mismatch");
	}
#endif
	return !skip_resample;
}

std::shared_ptr<ml_engine> ml_transformer::safe_get_ml_engine(ml_remapper_file::ml_remapper_types item)
{
	const auto engine = engines.find(item);
	if (engine != engines.end())
	{
		return engine->second;
	}
	const auto& settings = ml_remapper_file::ml_remappers.at(item);
	const auto& dimensions = settings.get_network_size();
	switch (item)
	{
	case ml_remapper_file::ml_remapper_types::off:
		break;
	case ml_remapper_file::ml_remapper_types::pass_through_test_engine:
	{
		engines.insert({ item, std::make_shared<dummy_pass_through_network>(dimensions.n()) });
	}
	break;
	case ml_remapper_file::ml_remapper_types::sperm_slim_40x:
		engines.insert({ item, std::make_shared<qli_unet_v1_semantic>(settings.network_resource_path,dimensions.width,dimensions.height,4) });;
		break;
	case ml_remapper_file::ml_remapper_types::hrslim_dapi_10x:
	case ml_remapper_file::ml_remapper_types::slim_dapi_10x:
	case ml_remapper_file::ml_remapper_types::glim_dapi_20x:
	case ml_remapper_file::ml_remapper_types::glim_dapi_20x_480:
	case ml_remapper_file::ml_remapper_types::glim_dil_20x:
	{
		engines.insert({ item, std::make_shared<qli_unet_v2_mapping>(settings.network_resource_path,dimensions.width,dimensions.height,settings.auxiliary_x1,settings.auxiliary_x2) });;
	}
	break;
	case ml_remapper_file::ml_remapper_types::dpm_slim:
		engines.insert({ item, std::make_shared<qli_unet_v3_mapping>(settings.network_resource_path,dimensions.width,dimensions.height,settings.auxiliary_x1,settings.auxiliary_x2) });;
		break;
	case ml_remapper_file::ml_remapper_types::viability:
		engines.insert({ item, std::make_shared<qli_ueffnet_v1_mapping>(settings.network_resource_path, dimensions.width, dimensions.height, settings.auxiliary_x1, settings.auxiliary_x2) });;
		break;
	default:
	{
		qli_runtime_error("Unsupported Mode");
	}
	}
	return engines.find(item)->second;
}

ml_transformer::file_to_engine_mapper ml_transformer::engines;


void ml_transformer::pre_bake()
{
	LOGGER_INFO("prebaking models.");
	for (const auto type : ml_remapper_file::mappers_to_prebake)
	{
		Q_UNUSED(safe_get_ml_engine(type));
	}
}
#else
void ml_transformer::pre_bake()
{

}
#endif

void ml_transformer::set_network_size(const frame_size& camera_frame_size)
{
	for (auto& change_me : ml_remapper_file::ml_remappers)
	{
		change_me.second.set_network_size(camera_frame_size);
	}
}

ml_transformer::ml_transformer(const frame_size& camera_frame_size)
{
	if (camera_frame_size.is_valid())
	{
		set_network_size(camera_frame_size);
	}
}

ml_transformer::~ml_transformer() = default;