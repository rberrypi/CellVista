#include "compute_engine_demosaic.h"
#include "thrust_resize.h"
#include "write_debug_gpu.h"
#include "write_tif.h"

//90 Formulas
#define roundfix(values) rintf(values)
#define get_90_90_A roundfix(foo(4, 4))
#define get_90_45_B roundfix(foo(4, 0)/96 - (3*foo(4, 2))/32 + (7*foo(4, 4))/12 + (7*foo(4, 6))/12 - (3*foo(4, 8))/32 + foo(4, 10)/96)
#define get_90_135_D roundfix(foo(0, 4)/96 - (3*foo(2, 4))/32 + (7*foo(4, 4))/12 + (7*foo(6, 4))/12 - (3*foo(8, 4))/32 + foo(10, 4)/96)
#define get_90_0_E roundfix(foo(0, 2)/1536 + foo(2, 0)/1536 + (7*foo(0, 4))/1536 - foo(2, 2)/128 + (7*foo(4, 0))/1536 + (7*foo(0, 6))/1536 - (61*foo(2, 4))/1536 - (61*foo(4, 2))/1536 + (7*foo(6, 0))/1536 + foo(0, 8)/1536 - (61*foo(2, 6))/1536 + (251*foo(4, 4))/768 - (61*foo(6, 2))/1536 + foo(8, 0)/1536 - foo(2, 8)/128 + (251*foo(4, 6))/768 + (251*foo(6, 4))/768 - foo(8, 2)/128 + foo(2, 10)/1536 - (61*foo(4, 8))/1536 + (251*foo(6, 6))/768 - (61*foo(8, 4))/1536 + foo(10, 2)/1536 + (7*foo(4, 10))/1536 - (61*foo(6, 8))/1536 - (61*foo(8, 6))/1536 + (7*foo(10, 4))/1536 + (7*foo(6, 10))/1536 - foo(8, 8)/128 + (7*foo(10, 6))/1536 + foo(8, 10)/1536 + foo(10, 8)/1536)
//45 Formulas
#define get_45_90_B roundfix(foo(4, -1)/96 - (3*foo(4, 1))/32 + (7*foo(4, 3))/12 + (7*foo(4, 5))/12 - (3*foo(4, 7))/32 + foo(4, 9)/96);
#define get_45_45_A roundfix(foo(4, 5))
#define get_45_135_E roundfix(foo(0, 1)/1536 + (7*foo(0, 3))/1536 + foo(2, -1)/1536 - foo(2, 1)/128 + (7*foo(0, 5))/1536 - (61*foo(2, 3))/1536 + (7*foo(4, -1))/1536 - (61*foo(4, 1))/1536 + foo(0, 7)/1536 - (61*foo(2, 5))/1536 + (251*foo(4, 3))/768 + (7*foo(6, -1))/1536 - (61*foo(6, 1))/1536 - foo(2, 7)/128 + (251*foo(4, 5))/768 + (251*foo(6, 3))/768 + foo(8, -1)/1536 - foo(8, 1)/128 + foo(2, 9)/1536 - (61*foo(4, 7))/1536 + (251*foo(6, 5))/768 - (61*foo(8, 3))/1536 + foo(10, 1)/1536 + (7*foo(4, 9))/1536 - (61*foo(6, 7))/1536 - (61*foo(8, 5))/1536 + (7*foo(10, 3))/1536 + (7*foo(6, 9))/1536 - foo(8, 7)/128 + (7*foo(10, 5))/1536 + foo(8, 9)/1536 + foo(10, 7)/1536)
#define get_45_0_D roundfix(foo(0, 5)/96 - (3*foo(2, 5))/32 + (7*foo(4, 5))/12 + (7*foo(6, 5))/12 - (3*foo(8, 5))/32 + foo(10, 5)/96)
//135 Formulas
#define get_135_90_D roundfix(foo(-1, 4)/96 - (3*foo(1, 4))/32 + (7*foo(3, 4))/12 + (7*foo(5, 4))/12 - (3*foo(7, 4))/32 + foo(9, 4)/96)
#define get_135_45_E roundfix(foo(1, 0)/1536 + foo(-1, 2)/1536 - foo(1, 2)/128 + (7*foo(3, 0))/1536 + (7*foo(-1, 4))/1536 - (61*foo(1, 4))/1536 - (61*foo(3, 2))/1536 + (7*foo(5, 0))/1536 + (7*foo(-1, 6))/1536 - (61*foo(1, 6))/1536 + (251*foo(3, 4))/768 - (61*foo(5, 2))/1536 + foo(7, 0)/1536 + foo(-1, 8)/1536 - foo(1, 8)/128 + (251*foo(3, 6))/768 + (251*foo(5, 4))/768 - foo(7, 2)/128 + foo(1, 10)/1536 - (61*foo(3, 8))/1536 + (251*foo(5, 6))/768 - (61*foo(7, 4))/1536 + foo(9, 2)/1536 + (7*foo(3, 10))/1536 - (61*foo(5, 8))/1536 - (61*foo(7, 6))/1536 + (7*foo(9, 4))/1536 + (7*foo(5, 10))/1536 - foo(7, 8)/128 + (7*foo(9, 6))/1536 + foo(7, 10)/1536 + foo(9, 8)/1536)
#define get_135_135_A roundfix(foo(5, 4))
#define get_135_0_B roundfix(foo(5, 0)/96 - (3*foo(5, 2))/32 + (7*foo(5, 4))/12 + (7*foo(5, 6))/12 - (3*foo(5, 8))/32 + foo(5, 10)/96)
//0 Formulas
#define get_0_90_E roundfix(foo(-1, 1)/1536 + foo(1, -1)/1536 - foo(1, 1)/128 + (7*foo(-1, 3))/1536 - (61*foo(1, 3))/1536 + (7*foo(3, -1))/1536 - (61*foo(3, 1))/1536 + (7*foo(-1, 5))/1536 - (61*foo(1, 5))/1536 + (251*foo(3, 3))/768 + (7*foo(5, -1))/1536 - (61*foo(5, 1))/1536 + foo(-1, 7)/1536 - foo(1, 7)/128 + (251*foo(3, 5))/768 + (251*foo(5, 3))/768 + foo(7, -1)/1536 - foo(7, 1)/128 + foo(1, 9)/1536 - (61*foo(3, 7))/1536 + (251*foo(5, 5))/768 - (61*foo(7, 3))/1536 + foo(9, 1)/1536 + (7*foo(3, 9))/1536 - (61*foo(5, 7))/1536 - (61*foo(7, 5))/1536 + (7*foo(9, 3))/1536 + (7*foo(5, 9))/1536 - foo(7, 7)/128 + (7*foo(9, 5))/1536 + foo(7, 9)/1536 + foo(9, 7)/1536)
#define get_0_45_D roundfix(foo(-1, 5)/96 - (3*foo(1, 5))/32 + (7*foo(3, 5))/12 + (7*foo(5, 5))/12 - (3*foo(7, 5))/32 + foo(9, 5)/96)
#define get_0_135_B roundfix(foo(5, -1)/96 - (3*foo(5, 1))/32 + (7*foo(5, 3))/12 + (7*foo(5, 5))/12 - (3*foo(5, 7))/32 + foo(5, 9)/96)
#define get_0_0_A roundfix(foo(5, 5))


template<class T>
__global__ void polarizer_demosaic_0_45_90_135(cudaTextureObject_t tex, T* output_a, T* output_b, T* output_c, T* output_d, const int rows, const int cols)
{
	const int col = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	const int row = 2 * ((blockIdx.y * blockDim.y) + threadIdx.y);
	if (col < cols && row < rows)
	{
		const auto out = [&](auto row, auto col)
		{
			return (row / 2) * (cols / 2) + (col / 2);
		};
#define foo(row_shift,col_shifts) (static_cast<float>(tex2D<unsigned short>(tex, col+(col_shifts-4), row+(row_shift-4))))
		output_a[out(row, col)] = get_90_90_A;
		output_b[out(row, col)] = get_45_90_B;
		output_c[out(row, col)] = get_135_90_D;
		output_d[out(row, col)] = get_0_90_E;
	}
}



template<typename T>
using demosaic_dual_function = void(*)(cudaTextureObject_t tex, T* output_a, T* output_d, int rows, int cols);

template<class T>
__global__ void polarizer_demosaic_45_135(cudaTextureObject_t tex, T* output_a, T* output_d, const int rows, const int cols)
{
	const int col = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	const int row = 2 * ((blockIdx.y * blockDim.y) + threadIdx.y);
	if (col < cols && row < rows)
	{
		const auto out = [&](auto row, auto col)
		{
			return (row / 2) * (cols / 2) + (col / 2);
		};
#define foo(row_shift,col_shifts) (static_cast<float>(tex2D<unsigned short>(tex, col+(col_shifts-4), row+(row_shift-4))))
		output_a[out(row, col)] = get_45_90_B;
		output_d[out(row, col)] = get_135_90_D;
	}
}

template<class T>
__global__ void polarizer_demosaic_0_90(cudaTextureObject_t tex, T* output_a, T* output_d, const int rows, const int cols)
{
	const int col = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	const int row = 2 * ((blockIdx.y * blockDim.y) + threadIdx.y);
	if (col < cols && row < rows)
	{
		const auto out = [&](auto row, auto col)
		{
			return (row / 2) * (cols / 2) + (col / 2);
		};
#define foo(row_shift,col_shifts) (static_cast<float>(tex2D<unsigned short>(tex, col+(col_shifts-4), row+(row_shift-4))))
		output_a[out(row, col)] = get_90_90_A;
		output_d[out(row, col)] = get_0_90_E;
	}
}

//0_45_90_135

template<class T, polarizer_demosaic_kind kind>
__global__ void polarizer_demosaic_grab_one(cudaTextureObject_t tex, T* output, const int rows, const int cols)
{
	const int col = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	const int row = 2 * ((blockIdx.y * blockDim.y) + threadIdx.y);
	if (col < cols && row < rows)
	{
		const auto out = [&](auto row, auto col)
		{
			return (row / 2) * (cols / 2) + (col / 2);
		};
#define foo(row_shift,col_shifts) (static_cast<float>(tex2D<unsigned short>(tex, col+(col_shifts-4), row+(row_shift-4))))
		switch (kind)
		{
		case polarizer_demosaic_kind::p90:
		{
			output[out(row, col)] = get_90_90_A;
		}
		break;
		case polarizer_demosaic_kind::p45:
		{
			output[out(row, col)] = get_45_90_B;
		}
		break;
		case polarizer_demosaic_kind::p135:
		{
			output[out(row, col)] = get_135_90_D;
		}
		break;
		case polarizer_demosaic_kind::p0:
		{
			output[out(row, col)] = get_0_90_E;
		}
		break;
		}
	}
}

frame_size demosaic_structs::demosaic_polarizer_pass_one(input_buffer& output_buffer_a, const camera_frame<unsigned short>& input_image_h, polarizer_demosaic_kind kind)
{
	const auto number_of_elements = input_image_h.samples();
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
	const auto block_size = 4;
	dim3 block(block_size, block_size, 1);
	auto const divide = [](auto numerator, auto denominator) {return static_cast<unsigned int>(ceil(numerator / (denominator * 1.0f))); };
	dim3 grid(divide(input_image_h.width / 2, block.x), divide(input_image_h.height / 2, block.y), 1);
	CUDASAFECALL(cudaMemcpy2D(demosaic_buffer_.res.pitch2D.devPtr, demosaic_buffer_.res.pitch2D.pitchInBytes, input_image_h.img, input_image_h.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.height, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_DEBUG_SYNC();
	//
	auto output_ptr_a = thrust_safe_get_pointer(output_buffer_a, number_of_elements / 4);
	const auto force_debug = false;
	switch (kind)
	{
	case polarizer_demosaic_kind::p0:
		polarizer_demosaic_grab_one<unsigned short, polarizer_demosaic_kind::p0> << <grid, block >> > (demosaic_buffer_tex_, output_ptr_a, input_image_h.height, input_image_h.width);
		break;
	case polarizer_demosaic_kind::p90:
		polarizer_demosaic_grab_one<unsigned short, polarizer_demosaic_kind::p90> << <grid, block >> > (demosaic_buffer_tex_, output_ptr_a, input_image_h.height, input_image_h.width);
		break;
	case polarizer_demosaic_kind::p45:
		polarizer_demosaic_grab_one<unsigned short, polarizer_demosaic_kind::p45> << <grid, block >> > (demosaic_buffer_tex_, output_ptr_a, input_image_h.height, input_image_h.width);
		break;
	case polarizer_demosaic_kind::p135:
		polarizer_demosaic_grab_one<unsigned short, polarizer_demosaic_kind::p135> << <grid, block >> > (demosaic_buffer_tex_, output_ptr_a, input_image_h.height, input_image_h.width);
		break;
	}
	write_debug_gpu(output_ptr_a, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_0_16.tif", force_debug);
	CUDA_DEBUG_SYNC();
	return  frame_size(input_image_h.width / 2, input_image_h.height / 2);
}

frame_size demosaic_structs::demosaic_polarizer_doubles(input_buffer& output_buffer_a, input_buffer& output_buffer_d, const camera_frame<unsigned short>& input_image_h, demosaic_mode mode)
{
	const auto number_of_elements = input_image_h.samples();
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
	const auto block_size = 4;
	dim3 block(block_size, block_size, 1);
	auto const divide = [](auto numerator, auto denominator) {return static_cast<unsigned int>(ceil(numerator / (denominator * 1.0f))); };
	dim3 grid(divide(input_image_h.width / 2, block.x), divide(input_image_h.height / 2, block.y), 1);
	CUDASAFECALL(cudaMemcpy2D(demosaic_buffer_.res.pitch2D.devPtr, demosaic_buffer_.res.pitch2D.pitchInBytes, input_image_h.img, input_image_h.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.height, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_DEBUG_SYNC();
	//
	auto output_ptr_a = thrust_safe_get_pointer(output_buffer_a, number_of_elements / 4);
	auto output_ptr_d = thrust_safe_get_pointer(output_buffer_d, number_of_elements / 4);
	const auto force_debug = false;
	demosaic_dual_function<unsigned short> function = [&]()
	{
		switch (mode)
		{
		case demosaic_mode::polarization_0_90:
			return polarizer_demosaic_0_90<unsigned short>;
		case demosaic_mode::polarization_45_135:
			return polarizer_demosaic_45_135<unsigned short>;
		default:
			qli_invalid_arguments();
		}
	}();
	function << <grid, block >> > (demosaic_buffer_tex_, output_ptr_a, output_ptr_d, input_image_h.height, input_image_h.width);
	write_debug_gpu(output_ptr_a, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_0_16.tif", force_debug);
	write_debug_gpu(output_ptr_d, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_1_16.tif", force_debug);
	CUDA_DEBUG_SYNC();
	return  frame_size(input_image_h.width / 2, input_image_h.height / 2);
}

frame_size demosaic_structs::demosaic_polarizer_0_45_90_135(input_buffer& output_buffer_a, input_buffer& output_buffer_b, input_buffer& output_buffer_c, input_buffer& output_buffer_d, const camera_frame<unsigned short>& input_image_h)
{
	const auto number_of_elements = input_image_h.samples();
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
	const auto block_size = 4;
	dim3 block(block_size, block_size, 1);
	auto const divide = [](auto numerator, auto denominator) {return static_cast<unsigned int>(ceil(numerator / (denominator * 1.0f))); };
	dim3 grid(divide(input_image_h.width / 2, block.x), divide(input_image_h.height / 2, block.y), 1);
	CUDASAFECALL(cudaMemcpy2D(demosaic_buffer_.res.pitch2D.devPtr, demosaic_buffer_.res.pitch2D.pitchInBytes, input_image_h.img, input_image_h.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.width * sizeof(unsigned short), demosaic_buffer_.res.pitch2D.height, cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_DEBUG_SYNC();
	//
	//Should go thorugh thrust_safe_get_pointer
	auto output_ptr_a = thrust_safe_get_pointer(output_buffer_a, number_of_elements / 4);
	auto output_ptr_b = thrust_safe_get_pointer(output_buffer_b, number_of_elements / 4);
	auto output_ptr_c = thrust_safe_get_pointer(output_buffer_c, number_of_elements / 4);
	auto output_ptr_d = thrust_safe_get_pointer(output_buffer_d, number_of_elements / 4);
	const auto force_debug = false;
	polarizer_demosaic_0_45_90_135 << <grid, block >> > (demosaic_buffer_tex_, output_ptr_a, output_ptr_b, output_ptr_c, output_ptr_d, input_image_h.height, input_image_h.width);
	write_debug_gpu(output_ptr_a, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_0_16.tif", force_debug);
	write_debug_gpu(output_ptr_b, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_1_16.tif", force_debug);
	write_debug_gpu(output_ptr_c, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_2_16.tif", force_debug);
	write_debug_gpu(output_ptr_d, input_image_h.width / 2, input_image_h.height / 2, 1, "polarizer_frame_3_16.tif", force_debug);
	CUDA_DEBUG_SYNC();
	return  frame_size(input_image_h.width / 2, input_image_h.height / 2);
}
