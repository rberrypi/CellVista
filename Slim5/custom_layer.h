#ifndef CUST_LAYER_H
#define CUST_LAYER_H
#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>



/*
	Customized Upsampling (Nearest Neighbor Interpolation) CUDA kernel
*/
inline __global__ void interpolation_resize_nearest_neighbor(float* out_arr, float const* in_arr, int in_h, int in_w, int in_c)
{
	// threadIdx.y tells the row, threadIdx.x tells the column
	// each thread should handle all the channel
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	const auto out_w = in_w * 2;
	const auto out_h = in_h * 2;
	//  Need to make sure this thread corresponds to a valid pixel on the original graph
	if (y < in_h && x < in_w) {
		//  Each thread is responsible for one pixel in all channels, [y][x] goes to [2y][2x]~[2y+1][2x+1]
		for (auto k = 0; k < in_c; k++) {
			auto color_to_copy = in_arr[k * in_w * in_h + in_w * y + x];
			out_arr[k * out_w * out_h + out_w * (2 * y) + 2 * x] = color_to_copy;
			out_arr[k * out_w * out_h + out_w * (2 * y + 1) + 2 * x] = color_to_copy;
			out_arr[k * out_w * out_h + out_w * (2 * y) + 2 * x + 1] = color_to_copy;
			out_arr[k * out_w * out_h + out_w * (2 * y + 1) + 2 * x + 1] = color_to_copy;
		}
	}
}


/*
	Customized Layer for Resize Nearest Neighbor
*/
class UpsampleBy2 : public nvinfer1::IPluginV2Ext
{
public:
	UpsampleBy2()
	{
		in_h = 0;
		in_w = 0;
		in_c = 0;
		mNamespace = "";
	}

	UpsampleBy2(int h, int w, int c)
	{
		in_h = h;
		in_w = w;
		in_c = c;
		mNamespace = "";
	}

	~UpsampleBy2()
	{

	}


	/*
		The builder checks the number of outputs and their dimensions using the
		following four plugin methods. Then the builder can connect the plugin
		layer to neighboring layers.
		The follow functions are used for this purpose.
	*/
	//  Used to specify number of output tensors
	[[nodiscard]] int getNbOutputs() const override
	{
		return 1;
	}

	//  Used to specify the dimensions of an output as a function of the input dimensions. 
	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1); //expecting only one input
		assert(inputs[0].nbDims == 4); //expecting the input to have NCHW
		//  N and C should stay the same, H and W are scaled 2 times
		return nvinfer1::DimsNCHW(inputs[0].d[0], inputs[0].d[1], 2 * inputs[0].d[2], 2 * inputs[0].d[3]);
	}

	//  Used to check if a plugin supports a given data format
	[[nodiscard]] bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
	{
		//  Our application should only expect FLOAT
		return ((type == nvinfer1::DataType::kFLOAT) && (format == nvinfer1::PluginFormat::kNCHW));
	}

	//  Used to get the data type of the output at a given index.The returned data type must have a format that is supported by the plugin
	nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override
	{
		return nvinfer1::DataType::kFLOAT;
	}

	/*
		The builder configures, initializes and executes the plugin at build time to discover
		optimal configurations.
		The following methods are used by the builder and the engine for this purpose.
	*/

	//  Communicates:
	//		the number of inputs and outputs, 
	//		dimensions and datatypes of all inputs and outputs, 
	//		broadcast information for all inputs and outputs, 
	//		the chosen plugin format, and maximum batch size. 
	//  At this point, the plugin sets up its internal state, and select the most appropriate algorithm and data structures for the given configuration. 
	void configurePlugin(
		const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
		const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,
		const bool* inputIsBroadcast, const bool* outputIsBroadcast,
		nvinfer1::PluginFormat floatFormat, int maxBatchSize) override
	{
		//  The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be 3-dimensional CHW dimensions). 
		//  When inputIsBroadcast or outputIsBroadcast is true, the outermost batch size for that input or output should be treated as if it is one. 
		//  inputIsBroadcast[i] is true only if the input is semantically broadcast across the batch and canBroadcastInputAcrossBatch(i) returned true. 
		//  outputIsBroadcast[i] is true only if isOutputBroadcastAcrossBatch(i) returned true. 

		//  TODO: don't understand what this function is supposed to achieve.
		return;
	}


	//  Initialize the layer for execution. This is called when the engine is created. 
	int initialize() override
	{
		//  TODO: don't understand what this function is supposed to achieve.
		return 0;
	}

	//  Execute the layer (Check LeakyRELU example on Github and sampleUffSSD
	int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		const auto safe_divide = [](auto a, auto b) {return static_cast<unsigned int>(ceil(a / b)); };
		dim3 block_dim(32, 32);
		dim3 grid_dim(safe_divide(in_h, block_dim.y), safe_divide(in_w, block_dim.x));
		//dim3 grid_dim(ceil(in_h / 32.0), ceil(in_w / 32.0));
		auto out_ptr = reinterpret_cast<float*>(outputs[0]);
		auto in_ptr = reinterpret_cast<float const*>(inputs[0]);
		interpolation_resize_nearest_neighbor << <grid_dim, block_dim, 0, stream >> > (out_ptr, in_ptr, in_h, in_w, in_c);
		return 0;
	}

	//  The engine context is destroyed and all the resources held by the plugin should be released. 
	void terminate() override
	{
		return;
	}

	[[nodiscard]] IPluginV2Ext* clone() const override
	{
		//  Copy h, w, c and namespace
		auto cloned = new UpsampleBy2(in_h, in_w, in_c);
		auto cloned_namespace = mNamespace.c_str();
		cloned->setPluginNamespace(cloned_namespace);
		return cloned;
	}

	void destroy() override
	{
		//  sample did "delete this;"
	}

	//  This method is used to set the library namespace that this plugin object belongs to(default can be "").
	//  All plugin objects from the same plugin library should have the same namespace.
	//  Following sampleUffSSD
	void setPluginNamespace(const char* pluginNamespace) override
	{
		this->mNamespace = pluginNamespace;
	}

	//  Following sampleUffSSD
	[[nodiscard]] const char* getPluginNamespace() const override
	{
		return this->mNamespace.c_str();
	}

	[[nodiscard]] bool canBroadcastInputAcrossBatch(int inputIndex) const override
	{
		return false;
	}

	bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override
	{
		return false;
	}

	//  Following sampleUffSSD
	[[nodiscard]] const char* getPluginType() const override
	{
		return "UpSampleBy2_TRT";
	}

	//  Following sampleUffSSD
	[[nodiscard]] const char* getPluginVersion() const override
	{
		return "1";
	}

	//  Following sampleUffSSD
	[[nodiscard]] size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	//  From https://github.com/LitLeo/TensorRT_Tutorial/blob/master/blogs/
	//  and sampleUffSSD
	//  these two functions just write the private attributes into a buffer
	[[nodiscard]] size_t getSerializationSize() const override
	{
		return sizeof(int) * 3;
	}

	void serialize(void* buffer) const override
	{
		*reinterpret_cast<int*>(buffer) = in_h;
		auto new_buffer = reinterpret_cast<int*>(buffer) + sizeof(int);
		*reinterpret_cast<int*>(new_buffer) = in_w;
		new_buffer += sizeof(int);
		*reinterpret_cast<int*>(new_buffer) = in_c;
	}

private:
	//  Parameters needed to calculate grid/block dimension
	int in_h;
	int in_w;
	int in_c;
	std::string mNamespace;
};







#endif

