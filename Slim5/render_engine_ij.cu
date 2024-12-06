#include "render_engine.h"
#include "thrust_resize.h"
#include "device_factory.h"
#include "image_j_pipe.h"
#include "cuda_error_check.h"

__global__ void back_to_binary(unsigned char* input_d, const int* input_d_img, const int numel)
{
	auto idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < numel)
	{
		input_d[idx] = input_d_img[idx] >= 0 ? 255 : 0;
	}
}

void render_engine::send_segmentation_d(const int* labels_d, const camera_frame<float>& img_d, const QString& full_path, const bool live)
{
	auto samples_per_pixel = img_d.samples_per_pixel;
	auto numel = img_d.n()*samples_per_pixel;
	static thrust::host_vector<float> img_h(numel);
	img_h.resize(numel);
	CUDASAFECALL(cudaMemcpy(img_h.data(), img_d.img, numel * sizeof(float), cudaMemcpyDeviceToHost));
	static thrust::device_vector<unsigned char> labels_d_binary;
	static int grid_size, block_size;
	{
		static auto old_size = 0;
		auto size_changed = (numel != old_size);
		if (size_changed)
		{
			int min_grid_size;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, back_to_binary, 0, 0));
			grid_size = (numel + block_size - 1) / block_size;
			old_size = numel;
		}
		auto labels_d_binary_ptr = thrust_safe_get_pointer(labels_d_binary, numel);
		back_to_binary << < grid_size, block_size >> > (labels_d_binary_ptr, labels_d, numel);
	}
	static thrust::host_vector<unsigned char> labels_h_binary;
	labels_h_binary.resize(numel);
	thrust::copy(labels_d_binary.begin(), labels_d_binary.end(), labels_h_binary.begin());
	//
	auto value = full_path.toStdString();
	D->ij->send_segmentation(labels_h_binary.data(), img_h.data(), img_d, img_d, img_d.render, value, live);
}