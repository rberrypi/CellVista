#include "haar_wavelet_gpu.h"
#include "thrust_resize.h"

__global__ void haarCD(float* buffer, const float* image, int W, int H)
{
	const auto i = threadIdx.x + blockIdx.x * blockDim.x;
	auto j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((j >= H / 2) || (i >= W / 2))
	{
		return;
	}
	const auto co = 2 * (i);
	const auto ro = 2 * (j);
	const auto in = ro * W + co;
	const auto a = image[in + 0 + 0];
	const auto b = image[in + 1 + 0];
	const auto c = image[in + 0 + W];
	const auto d = image[in + 1 + W];
	const auto value = ((a + d) - (b + c)) * 0.5f;
	buffer[i + j * (W / 2)] = value;
}

__global__ void haarCH(float* buffer, const float* image, int W, int H)
{
	const auto i = threadIdx.x + blockIdx.x * blockDim.x;
	const auto j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((j >= H / 2) || (i >= W / 2))
	{
		return;
	}
	auto co = 2 * (i);
	auto ro = 2 * (j);
	auto in = ro * W + co;
	const auto a = image[in + 0 + 0];
	const auto b = image[in + 1 + 0];
	const auto c = image[in + 0 + W];
	const auto d = image[in + 1 + W];
	const auto value = ((a + b) - (c + d)) * 0.5f;
	buffer[i + j * (W / 2)] = value;
}

__global__ void haarCV(float* buffer, const float* image, int W, int H)
{
	const auto i = threadIdx.x + blockIdx.x * blockDim.x;
	const auto j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((j >= H / 2) || (i >= W / 2))
	{
		return;
	}
	const auto co = 2 * (i);
	const auto ro = 2 * (j);
	const auto in = ro * W + co;
	const auto a = image[in + 0 + 0];
	const auto b = image[in + 1 + 0];
	const auto c = image[in + 0 + W];
	const auto d = image[in + 1 + W];
	const auto value = ((a + c) - (b + d)) * 0.5f;
	buffer[i + j * (W / 2)] = value;
}

__global__ void haarCA(float* buffer, const float* image, int W, int H)
{
	const auto i = threadIdx.x + blockIdx.x * blockDim.x;
	const auto j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((j >= H / 2) || (i >= W / 2))
	{
		return;
	}
	const auto co = 2 * (i);
	const auto ro = 2 * (j);
	const auto in = ro * W + co;
	const auto a = image[in + 0 + 0];
	const auto b = image[in + 1 + 0];
	const auto c = image[in + 0 + W];
	const auto d = image[in + 1 + W];
	const auto value = ((a + c) + (b + d)) * 0.5f;
	buffer[i + j * (W / 2)] = value;
}

void haar_wavelet_gpu::haarlet(thrust::device_vector<float>& outputD_vec, const float* inputD, const frame_size& s, bool low_pass)
{
	auto output_d = thrust_safe_get_pointer(outputD_vec, s.n());
	const auto W = s.width;
	const auto H = s.height;
	const auto num_elements_x = W / 2;//some random guess, guess first profile later
	const auto num_elements_y = H / 2;
	dim3 block_size;
	block_size.x = 16;//2544x2160
	block_size.y = 16;
	dim3 grid_size;
	grid_size.x = static_cast<unsigned int>(ceil(num_elements_x / (1.f * block_size.x)));
	grid_size.y = static_cast<unsigned int>(ceil(num_elements_y / (1.f * block_size.y)));
	const auto sizeofit = W * H / 4;
	{
		auto part_d = &output_d[sizeofit * (0)];
		haarCD << <grid_size, block_size >> > (part_d, inputD, W, H);
	}
	{
		auto part_d = &output_d[sizeofit * (1)];
		haarCH << <grid_size, block_size >> > (part_d, inputD, W, H);
	}
	{
		auto part_d = &output_d[sizeofit * (2)];
		haarCV << <grid_size, block_size >> > (part_d, inputD, W, H);
	}
	if (low_pass) //often not needed
	{
		auto part_d = &output_d[sizeofit * (3)];
		haarCA << <grid_size, block_size >> > (part_d, inputD, W, H);
	}
}

float haar_wavelet_gpu::compute_fusion_focus(const camera_frame<float>& input)
{
	return compute_fusion_focus(input.img, input);
}

float haar_wavelet_gpu::compute_fusion_focus(const float* input, const frame_size& size)
{
	// todo is this kernel in nvpp?
	// todo assert divisible by four?
	haarlet(haar_result_, input, size, true);
	auto high = 0.0f;
	auto low = 0.0f;
	const auto small_n = size.n() / 4;
	for (auto i = 0; i < 4; i++)
	{
		//small_n*i
		auto start = haar_result_.begin() + small_n * i;
		auto stop = haar_result_.begin() + small_n * (1 + i);
		auto variance = get_variance(start, stop);
		if (i == 3)//wierd
		{
			low = variance;
		}
		else
		{
			high += variance;
		}
	}
	return high * low;
}
