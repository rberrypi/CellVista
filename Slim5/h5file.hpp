#pragma once
#ifndef H5_FILE_H
#define H5_FILE_H
#include "ml_shared.h"
#define H5_BUILT_AS_DYNAMIC_LIB
#include <vector>
#include <string>
#include <H5Cpp.h>
#include <iostream>
/*
szip.lib
zlib.lib
hdf5.lib
hdf5_cpp.lib
libtiff.lib
 */
#pragma comment(lib, "hdf5.lib")
#pragma comment(lib, "hdf5_cpp.lib")


static std::vector<float> load_weight_new(
	const char* file_name, const char* group_name_1, const char* group_name_2,
	const char* ds_name, const int count)
{
	//  Open the file
	H5::H5File fp(file_name, H5F_ACC_RDONLY);
	const auto dataset = fp.openGroup(group_name_1).openGroup(group_name_2).openDataSet(ds_name);
	//  Checking: make sure the number of weight values is correct
	const auto size = dataset.getInMemDataSize() / sizeof(float);
	assert(size == count);
	//  Read the values
	std::vector<float> weights(size, (333333.3f));
	dataset.read(weights.data(), H5::PredType::NATIVE_FLOAT);
	fp.close();
	return weights;
}

void  attr_op(H5::H5Location& loc, const std::string attr_name, void* operator_data)
{
	std::cout << attr_name << std::endl;
}

static std::vector<float> load_weight(const char* file_name, const char* layer_name, const char* type, const int count)
{
	//  Open the file
	H5::H5File fp(file_name, H5F_ACC_RDONLY);
	const auto dataset = fp.openGroup(layer_name).openGroup(layer_name).openDataSet(type);
	//  Checking: make sure the number of weight values is correct
	const auto size = dataset.getInMemDataSize() / sizeof(float);
	//  Read the values
	std::vector<float> weights(size, (333333.3f));
	dataset.read(weights.data(), H5::PredType::NATIVE_FLOAT);
	fp.close();
	return weights;
}

static void load_these_weights(const std::string& weight_file_name)
{

	H5::H5File fp(weight_file_name.c_str(), H5F_ACC_RDONLY, H5::FileCreatPropList::DEFAULT, H5::FileAccPropList::DEFAULT);

	//  From the Python API, the name "conv_24" can be used to identify a Group
	auto group = fp.openGroup("conv2d_24");
	auto smaller_group = group.openGroup("conv2d_24");
	const auto ds_bias = smaller_group.openDataSet("bias:0");
	const auto ds_kernel = smaller_group.openDataSet("kernel:0");
	const auto size_bias_Test = ds_bias.getStorageSize();
	const auto size_bias = ds_bias.getInMemDataSize();
	const auto size_kernel = ds_kernel.getInMemDataSize();
	std::vector<double> kernel(3 * 3 * 1 * 64, (0.33f));

	for (auto k = 0; k < 3 * 3 * 1 * 64; ++k)
	{
		std::cout << kernel[k] << std::endl;
	}
	ds_kernel.read(kernel.data(), H5::PredType::NATIVE_DOUBLE);
	std::cout << kernel.size() << std::endl;
	for (auto k = 0; k < 3 * 3 * 1 * 64; ++k)
	{
		std::cout << kernel[k] << std::endl;
	}
	fp.close();
}

//  Check if anything wrong with weight loading
static void massive_weight_loading_test()
{
	std::vector<std::string> weight_names = {
		"conv2d_24", "conv2d_25", "conv2d_26", "conv2d_27",
		"conv2d_28", "conv2d_29", "conv2d_30", "conv2d_31",
		"conv2d_32", "conv2d_33", "conv2d_34", "conv2d_35",
		"conv2d_36", "conv2d_37", "conv2d_38", "conv2d_39",
		"conv2d_40", "conv2d_41", "conv2d_42", "conv2d_43",
		"conv2d_44", "conv2d_45", "conv2d_46"
	};


	return;
}

#endif