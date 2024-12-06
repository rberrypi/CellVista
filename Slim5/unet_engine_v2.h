#pragma once
#ifndef FL_MODEL_H
#define FL_MODEL_H

#include <NvInfer.h>
#include <cassert>
#include "h5file.hpp"
#include "ml_shared.h"
#include "qli_runtime_error.h"

nvinfer1::ICudaEngine* create_unet_engine_v2(nvinfer1::IBuilder* builder, const char* weight_fname, int in_c, int in_h, int in_w, float x_min, float x_max)
{
	// it doesn't make sense to have 0 width and height image
	if (!is_divisible_by_sixteen_nonzero(in_h) || !is_divisible_by_sixteen_nonzero(in_w))
	{
		qli_invalid_arguments();
	}
	//  Create the network to populate the network, then set the outputs and create an engine
	auto network = builder->createNetwork();

	//  Add layers to the network
	auto in_tensor = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::DimsNCHW(1, in_c, in_h, in_w));
	assert(in_tensor != nullptr);


	/*
	 * Scaling the input image to [0,1]
	 * For Dr.Sobh's model, we need:
	 * magic_x_min = -3.1446402
	 * magic_x_max = 2.2227864
	 */


	std::vector<float> shift_value{ -x_min / (x_max - x_min) };
	nvinfer1::Weights input_shift{ nvinfer1::DataType::kFLOAT, shift_value.data(), 1 };
	std::vector<float> scale_value{ 1 / (x_max - x_min) };
	nvinfer1::Weights input_scale{ nvinfer1::DataType::kFLOAT, scale_value.data(), 1 };
	std::vector<float> power_value{ 1 };
	nvinfer1::Weights input_power{ nvinfer1::DataType::kFLOAT, power_value.data(), 1 };
	auto scaled_input = network->addScale(*in_tensor, nvinfer1::ScaleMode::kUNIFORM, input_shift, input_scale, input_power);
	assert(scaled_input != nullptr);

	/* ------------	*/
	/*	Block 1		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_1_kernel = load_weight_new(weight_fname, "conv2d_1", "conv2d_1_1", "kernel:0", 3 * 3 * 1 * 16);
	auto conv_1_bias = load_weight_new(weight_fname, "conv2d_1", "conv2d_1_1", "bias:0", 16);
	//  Convert to TRT format
	nvinfer1::Weights conv_1_kernel_w{ nvinfer1::DataType::kFLOAT, conv_1_kernel.data(), 3 * 3 * 1 * 16 };
	nvinfer1::Weights conv_1_bias_w{ nvinfer1::DataType::kFLOAT, conv_1_bias.data(), 16 };
	//  Add conv
	auto pad_1 = network->addPadding(*(scaled_input->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_1 != nullptr);
	auto conv_1 = network->addConvolution(*(pad_1->getOutput(0)), 16, nvinfer1::DimsHW(3, 3), conv_1_kernel_w, conv_1_bias_w);
	assert(conv_1 != nullptr);
	//  Add BN
	auto bn_1_layer_name_1 = "batch_normalization_1";
	auto bn_1_layer_name_2 = "batch_normalization_1_1";
	auto bn_1_filter_count = 16;
	auto bn_1_beta = load_weight_new(weight_fname, bn_1_layer_name_1, bn_1_layer_name_2, "beta:0", bn_1_filter_count);
	auto bn_1_gamma = load_weight_new(weight_fname, bn_1_layer_name_1, bn_1_layer_name_2, "gamma:0", bn_1_filter_count);
	auto bn_1_mean = load_weight_new(weight_fname, bn_1_layer_name_1, bn_1_layer_name_2, "moving_mean:0", bn_1_filter_count);
	auto bn_1_variance = load_weight_new(weight_fname, bn_1_layer_name_1, bn_1_layer_name_2, "moving_variance:0", bn_1_filter_count);

	std::vector<float> bn_1_shift_v(bn_1_filter_count, 0.0);
	std::vector<float> bn_1_scale_v(bn_1_filter_count, 0.0);
	std::vector<float> bn_1_power_v(bn_1_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_1_filter_count; ++c_idx)
	{
		bn_1_shift_v[c_idx] = bn_1_beta[c_idx] - (bn_1_gamma[c_idx] * bn_1_mean[c_idx] / sqrtf(bn_1_variance[c_idx] + 0.001));
		bn_1_scale_v[c_idx] = bn_1_gamma[c_idx] / sqrtf(bn_1_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_1_shift_w{ nvinfer1::DataType::kFLOAT, bn_1_shift_v.data(), bn_1_filter_count };
	nvinfer1::Weights bn_1_scale_w{ nvinfer1::DataType::kFLOAT, bn_1_scale_v.data(), bn_1_filter_count };
	nvinfer1::Weights bn_1_power_w{ nvinfer1::DataType::kFLOAT, bn_1_power_v.data(), bn_1_filter_count };
	auto bn_1 = network->addScale(*(conv_1->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_1_shift_w, bn_1_scale_w, bn_1_power_w);

	//  Add Activation
	auto relu_1 = network->addActivation(*(bn_1->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 2		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_2_kernel = load_weight_new(weight_fname, "conv2d_2", "conv2d_2_1", "kernel:0", 3 * 3 * 16 * 16);
	auto conv_2_bias = load_weight_new(weight_fname, "conv2d_2", "conv2d_2_1", "bias:0", 16);
	//  Convert to TRT format
	nvinfer1::Weights conv_2_kernel_w{ nvinfer1::DataType::kFLOAT, conv_2_kernel.data(), 3 * 3 * 16 * 16 };
	nvinfer1::Weights conv_2_bias_w{ nvinfer1::DataType::kFLOAT, conv_2_bias.data(), 16 };
	//  Add conv
	auto pad_2 = network->addPadding(*(relu_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_2 != nullptr);
	auto conv_2 = network->addConvolution(*(pad_2->getOutput(0)), 16, nvinfer1::DimsHW(3, 3), conv_2_kernel_w, conv_2_bias_w);
	assert(conv_2 != nullptr);
	//  Add BN
	auto bn_2_layer_name_1 = "batch_normalization_2";
	auto bn_2_layer_name_2 = "batch_normalization_2_1";
	auto bn_2_filter_count = 16;
	auto bn_2_beta = load_weight_new(weight_fname, bn_2_layer_name_1, bn_2_layer_name_2, "beta:0", bn_2_filter_count);
	auto bn_2_gamma = load_weight_new(weight_fname, bn_2_layer_name_1, bn_2_layer_name_2, "gamma:0", bn_2_filter_count);
	auto bn_2_mean = load_weight_new(weight_fname, bn_2_layer_name_1, bn_2_layer_name_2, "moving_mean:0", bn_2_filter_count);
	auto bn_2_variance = load_weight_new(weight_fname, bn_2_layer_name_1, bn_2_layer_name_2, "moving_variance:0", bn_2_filter_count);

	std::vector<float> bn_2_shift_v(bn_2_filter_count, 0.0);
	std::vector<float> bn_2_scale_v(bn_2_filter_count, 0.0);
	std::vector<float> bn_2_power_v(bn_2_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_2_filter_count; ++c_idx)
	{
		bn_2_shift_v[c_idx] = bn_2_beta[c_idx] - (bn_2_gamma[c_idx] * bn_2_mean[c_idx] / sqrtf(bn_2_variance[c_idx] + 0.001));
		bn_2_scale_v[c_idx] = bn_2_gamma[c_idx] / sqrtf(bn_2_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_2_shift_w{ nvinfer1::DataType::kFLOAT, bn_2_shift_v.data(), bn_2_filter_count };
	nvinfer1::Weights bn_2_scale_w{ nvinfer1::DataType::kFLOAT, bn_2_scale_v.data(), bn_2_filter_count };
	nvinfer1::Weights bn_2_power_w{ nvinfer1::DataType::kFLOAT, bn_2_power_v.data(), bn_2_filter_count };
	auto bn_2 = network->addScale(*(conv_2->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_2_shift_w, bn_2_scale_w, bn_2_power_w);

	//  Add Activation
	auto relu_2 = network->addActivation(*(bn_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_1 = network->addPooling(*(relu_2->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));


	/* ------------	*/
	/*	Block 3		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_3_kernel = load_weight_new(weight_fname, "conv2d_3", "conv2d_3_1", "kernel:0", 3 * 3 * 16 * 32);
	auto conv_3_bias = load_weight_new(weight_fname, "conv2d_3", "conv2d_3_1", "bias:0", 32);
	//  Convert to TRT format
	nvinfer1::Weights conv_3_kernel_w{ nvinfer1::DataType::kFLOAT, conv_3_kernel.data(), 3 * 3 * 16 * 32 };
	nvinfer1::Weights conv_3_bias_w{ nvinfer1::DataType::kFLOAT, conv_3_bias.data(), 32 };
	//  Add conv
	auto pad_3 = network->addPadding(*(pool_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_3 != nullptr);
	auto conv_3 = network->addConvolution(*(pad_3->getOutput(0)), 32, nvinfer1::DimsHW(3, 3), conv_3_kernel_w, conv_3_bias_w);
	assert(conv_3 != nullptr);
	//  Add BN
	auto bn_3_layer_name_1 = "batch_normalization_3";
	auto bn_3_layer_name_2 = "batch_normalization_3_1";
	auto bn_3_filter_count = 32;
	auto bn_3_beta = load_weight_new(weight_fname, bn_3_layer_name_1, bn_3_layer_name_2, "beta:0", bn_3_filter_count);
	auto bn_3_gamma = load_weight_new(weight_fname, bn_3_layer_name_1, bn_3_layer_name_2, "gamma:0", bn_3_filter_count);
	auto bn_3_mean = load_weight_new(weight_fname, bn_3_layer_name_1, bn_3_layer_name_2, "moving_mean:0", bn_3_filter_count);
	auto bn_3_variance = load_weight_new(weight_fname, bn_3_layer_name_1, bn_3_layer_name_2, "moving_variance:0", bn_3_filter_count);

	std::vector<float> bn_3_shift_v(bn_3_filter_count, 0.0);
	std::vector<float> bn_3_scale_v(bn_3_filter_count, 0.0);
	std::vector<float> bn_3_power_v(bn_3_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_3_filter_count; ++c_idx)
	{
		bn_3_shift_v[c_idx] = bn_3_beta[c_idx] - (bn_3_gamma[c_idx] * bn_3_mean[c_idx] / sqrtf(bn_3_variance[c_idx] + 0.001));
		bn_3_scale_v[c_idx] = bn_3_gamma[c_idx] / sqrtf(bn_3_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_3_shift_w{ nvinfer1::DataType::kFLOAT, bn_3_shift_v.data(), bn_3_filter_count };
	nvinfer1::Weights bn_3_scale_w{ nvinfer1::DataType::kFLOAT, bn_3_scale_v.data(), bn_3_filter_count };
	nvinfer1::Weights bn_3_power_w{ nvinfer1::DataType::kFLOAT, bn_3_power_v.data(), bn_3_filter_count };
	auto bn_3 = network->addScale(*(conv_3->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_3_shift_w, bn_3_scale_w, bn_3_power_w);

	//  Add Activation
	auto relu_3 = network->addActivation(*(bn_3->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 4		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_4_kernel = load_weight_new(weight_fname, "conv2d_4", "conv2d_4_1", "kernel:0", 3 * 3 * 32 * 32);
	auto conv_4_bias = load_weight_new(weight_fname, "conv2d_4", "conv2d_4_1", "bias:0", 32);
	//  Convert to TRT format
	nvinfer1::Weights conv_4_kernel_w{ nvinfer1::DataType::kFLOAT, conv_4_kernel.data(), 3 * 3 * 32 * 32 };
	nvinfer1::Weights conv_4_bias_w{ nvinfer1::DataType::kFLOAT, conv_4_bias.data(), 32 };
	//  Add conv
	auto pad_4 = network->addPadding(*(relu_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_4 != nullptr);
	auto conv_4 = network->addConvolution(*(pad_4->getOutput(0)), 32, nvinfer1::DimsHW(3, 3), conv_4_kernel_w, conv_4_bias_w);
	assert(conv_4 != nullptr);
	//  Add BN
	auto bn_4_layer_name_1 = "batch_normalization_4";
	auto bn_4_layer_name_2 = "batch_normalization_4_1";
	auto bn_4_filter_count = 32;
	auto bn_4_beta = load_weight_new(weight_fname, bn_4_layer_name_1, bn_4_layer_name_2, "beta:0", bn_4_filter_count);
	auto bn_4_gamma = load_weight_new(weight_fname, bn_4_layer_name_1, bn_4_layer_name_2, "gamma:0", bn_4_filter_count);
	auto bn_4_mean = load_weight_new(weight_fname, bn_4_layer_name_1, bn_4_layer_name_2, "moving_mean:0", bn_4_filter_count);
	auto bn_4_variance = load_weight_new(weight_fname, bn_4_layer_name_1, bn_4_layer_name_2, "moving_variance:0", bn_4_filter_count);

	std::vector<float> bn_4_shift_v(bn_4_filter_count, 0.0);
	std::vector<float> bn_4_scale_v(bn_4_filter_count, 0.0);
	std::vector<float> bn_4_power_v(bn_4_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_4_filter_count; ++c_idx)
	{
		bn_4_shift_v[c_idx] = bn_4_beta[c_idx] - (bn_4_gamma[c_idx] * bn_4_mean[c_idx] / sqrtf(bn_4_variance[c_idx] + 0.001));
		bn_4_scale_v[c_idx] = bn_4_gamma[c_idx] / sqrtf(bn_4_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_4_shift_w{ nvinfer1::DataType::kFLOAT, bn_4_shift_v.data(), bn_4_filter_count };
	nvinfer1::Weights bn_4_scale_w{ nvinfer1::DataType::kFLOAT, bn_4_scale_v.data(), bn_4_filter_count };
	nvinfer1::Weights bn_4_power_w{ nvinfer1::DataType::kFLOAT, bn_4_power_v.data(), bn_4_filter_count };
	auto bn_4 = network->addScale(*(conv_4->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_4_shift_w, bn_4_scale_w, bn_4_power_w);

	//  Add Activation
	auto relu_4 = network->addActivation(*(bn_4->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_2 = network->addPooling(*(relu_4->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));

	/* ------------	*/
	/*	Block 5		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_5_kernel = load_weight_new(weight_fname, "conv2d_5", "conv2d_5_1", "kernel:0", 3 * 3 * 32 * 64);
	auto conv_5_bias = load_weight_new(weight_fname, "conv2d_5", "conv2d_5_1", "bias:0", 64);
	//  Convert to TRT format
	nvinfer1::Weights conv_5_kernel_w{ nvinfer1::DataType::kFLOAT, conv_5_kernel.data(), 3 * 3 * 32 * 64 };
	nvinfer1::Weights conv_5_bias_w{ nvinfer1::DataType::kFLOAT, conv_5_bias.data(), 64 };
	//  Add conv
	auto pad_5 = network->addPadding(*(pool_2->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_5 != nullptr);
	auto conv_5 = network->addConvolution(*(pad_5->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), conv_5_kernel_w, conv_5_bias_w);
	assert(conv_5 != nullptr);
	//  Add BN
	auto bn_5_layer_name_1 = "batch_normalization_5";
	auto bn_5_layer_name_2 = "batch_normalization_5_1";
	auto bn_5_filter_count = 64;
	auto bn_5_beta = load_weight_new(weight_fname, bn_5_layer_name_1, bn_5_layer_name_2, "beta:0", bn_5_filter_count);
	auto bn_5_gamma = load_weight_new(weight_fname, bn_5_layer_name_1, bn_5_layer_name_2, "gamma:0", bn_5_filter_count);
	auto bn_5_mean = load_weight_new(weight_fname, bn_5_layer_name_1, bn_5_layer_name_2, "moving_mean:0", bn_5_filter_count);
	auto bn_5_variance = load_weight_new(weight_fname, bn_5_layer_name_1, bn_5_layer_name_2, "moving_variance:0", bn_5_filter_count);

	std::vector<float> bn_5_shift_v(bn_5_filter_count, 0.0);
	std::vector<float> bn_5_scale_v(bn_5_filter_count, 0.0);
	std::vector<float> bn_5_power_v(bn_5_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_5_filter_count; ++c_idx)
	{
		bn_5_shift_v[c_idx] = bn_5_beta[c_idx] - (bn_5_gamma[c_idx] * bn_5_mean[c_idx] / sqrtf(bn_5_variance[c_idx] + 0.001));
		bn_5_scale_v[c_idx] = bn_5_gamma[c_idx] / sqrtf(bn_5_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_5_shift_w{ nvinfer1::DataType::kFLOAT, bn_5_shift_v.data(), bn_5_filter_count };
	nvinfer1::Weights bn_5_scale_w{ nvinfer1::DataType::kFLOAT, bn_5_scale_v.data(), bn_5_filter_count };
	nvinfer1::Weights bn_5_power_w{ nvinfer1::DataType::kFLOAT, bn_5_power_v.data(), bn_5_filter_count };
	auto bn_5 = network->addScale(*(conv_5->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_5_shift_w, bn_5_scale_w, bn_5_power_w);

	//  Add Activation
	auto relu_5 = network->addActivation(*(bn_5->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 6		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_6_kernel = load_weight_new(weight_fname, "conv2d_6", "conv2d_6_1", "kernel:0", 3 * 3 * 64 * 64);
	auto conv_6_bias = load_weight_new(weight_fname, "conv2d_6", "conv2d_6_1", "bias:0", 64);
	//  Convert to TRT format
	nvinfer1::Weights conv_6_kernel_w{ nvinfer1::DataType::kFLOAT, conv_6_kernel.data(), 3 * 3 * 64 * 64 };
	nvinfer1::Weights conv_6_bias_w{ nvinfer1::DataType::kFLOAT, conv_6_bias.data(), 64 };
	//  Add conv
	auto pad_6 = network->addPadding(*(relu_5->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_6 != nullptr);
	auto conv_6 = network->addConvolution(*(pad_6->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), conv_6_kernel_w, conv_6_bias_w);
	assert(conv_6 != nullptr);
	//  Add BN
	auto bn_6_layer_name_1 = "batch_normalization_6";
	auto bn_6_layer_name_2 = "batch_normalization_6_1";
	auto bn_6_filter_count = 64;
	auto bn_6_beta = load_weight_new(weight_fname, bn_6_layer_name_1, bn_6_layer_name_2, "beta:0", bn_6_filter_count);
	auto bn_6_gamma = load_weight_new(weight_fname, bn_6_layer_name_1, bn_6_layer_name_2, "gamma:0", bn_6_filter_count);
	auto bn_6_mean = load_weight_new(weight_fname, bn_6_layer_name_1, bn_6_layer_name_2, "moving_mean:0", bn_6_filter_count);
	auto bn_6_variance = load_weight_new(weight_fname, bn_6_layer_name_1, bn_6_layer_name_2, "moving_variance:0", bn_6_filter_count);

	std::vector<float> bn_6_shift_v(bn_6_filter_count, 0.0);
	std::vector<float> bn_6_scale_v(bn_6_filter_count, 0.0);
	std::vector<float> bn_6_power_v(bn_6_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_6_filter_count; ++c_idx)
	{
		bn_6_shift_v[c_idx] = bn_6_beta[c_idx] - (bn_6_gamma[c_idx] * bn_6_mean[c_idx] / sqrtf(bn_6_variance[c_idx] + 0.001));
		bn_6_scale_v[c_idx] = bn_6_gamma[c_idx] / sqrtf(bn_6_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_6_shift_w{ nvinfer1::DataType::kFLOAT, bn_6_shift_v.data(), bn_6_filter_count };
	nvinfer1::Weights bn_6_scale_w{ nvinfer1::DataType::kFLOAT, bn_6_scale_v.data(), bn_6_filter_count };
	nvinfer1::Weights bn_6_power_w{ nvinfer1::DataType::kFLOAT, bn_6_power_v.data(), bn_6_filter_count };
	auto bn_6 = network->addScale(*(conv_6->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_6_shift_w, bn_6_scale_w, bn_6_power_w);

	//  Add Activation
	auto relu_6 = network->addActivation(*(bn_6->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_3 = network->addPooling(*(relu_6->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));

	/* ------------	*/
	/*	Block 7		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_7_kernel = load_weight_new(weight_fname, "conv2d_7", "conv2d_7_1", "kernel:0", 3 * 3 * 64 * 128);
	auto conv_7_bias = load_weight_new(weight_fname, "conv2d_7", "conv2d_7_1", "bias:0", 128);
	//  Convert to TRT format
	nvinfer1::Weights conv_7_kernel_w{ nvinfer1::DataType::kFLOAT, conv_7_kernel.data(), 3 * 3 * 64 * 128 };
	nvinfer1::Weights conv_7_bias_w{ nvinfer1::DataType::kFLOAT, conv_7_bias.data(), 128 };
	//  Add conv
	auto pad_7 = network->addPadding(*(pool_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_7 != nullptr);
	auto conv_7 = network->addConvolution(*(pad_7->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), conv_7_kernel_w, conv_7_bias_w);
	assert(conv_7 != nullptr);
	//  Add BN
	auto bn_7_layer_name_1 = "batch_normalization_7";
	auto bn_7_layer_name_2 = "batch_normalization_7_1";
	auto bn_7_filter_count = 128;
	auto bn_7_beta = load_weight_new(weight_fname, bn_7_layer_name_1, bn_7_layer_name_2, "beta:0", bn_7_filter_count);
	auto bn_7_gamma = load_weight_new(weight_fname, bn_7_layer_name_1, bn_7_layer_name_2, "gamma:0", bn_7_filter_count);
	auto bn_7_mean = load_weight_new(weight_fname, bn_7_layer_name_1, bn_7_layer_name_2, "moving_mean:0", bn_7_filter_count);
	auto bn_7_variance = load_weight_new(weight_fname, bn_7_layer_name_1, bn_7_layer_name_2, "moving_variance:0", bn_7_filter_count);

	std::vector<float> bn_7_shift_v(bn_7_filter_count, 0.0);
	std::vector<float> bn_7_scale_v(bn_7_filter_count, 0.0);
	std::vector<float> bn_7_power_v(bn_7_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_7_filter_count; ++c_idx)
	{
		bn_7_shift_v[c_idx] = bn_7_beta[c_idx] - (bn_7_gamma[c_idx] * bn_7_mean[c_idx] / sqrtf(bn_7_variance[c_idx] + 0.001));
		bn_7_scale_v[c_idx] = bn_7_gamma[c_idx] / sqrtf(bn_7_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_7_shift_w{ nvinfer1::DataType::kFLOAT, bn_7_shift_v.data(), bn_7_filter_count };
	nvinfer1::Weights bn_7_scale_w{ nvinfer1::DataType::kFLOAT, bn_7_scale_v.data(), bn_7_filter_count };
	nvinfer1::Weights bn_7_power_w{ nvinfer1::DataType::kFLOAT, bn_7_power_v.data(), bn_7_filter_count };
	auto bn_7 = network->addScale(*(conv_7->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_7_shift_w, bn_7_scale_w, bn_7_power_w);

	//  Add Activation
	auto relu_7 = network->addActivation(*(bn_7->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 8		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_8_kernel = load_weight_new(weight_fname, "conv2d_8", "conv2d_8_1", "kernel:0", 3 * 3 * 128 * 128);
	auto conv_8_bias = load_weight_new(weight_fname, "conv2d_8", "conv2d_8_1", "bias:0", 128);
	//  Convert to TRT format
	nvinfer1::Weights conv_8_kernel_w{ nvinfer1::DataType::kFLOAT, conv_8_kernel.data(), 3 * 3 * 128 * 128 };
	nvinfer1::Weights conv_8_bias_w{ nvinfer1::DataType::kFLOAT, conv_8_bias.data(), 128 };
	//  Add conv
	auto pad_8 = network->addPadding(*(relu_7->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_8 != nullptr);
	auto conv_8 = network->addConvolution(*(pad_8->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), conv_8_kernel_w, conv_8_bias_w);
	assert(conv_8 != nullptr);
	//  Add BN
	auto bn_8_layer_name_1 = "batch_normalization_8";
	auto bn_8_layer_name_2 = "batch_normalization_8_1";
	auto bn_8_filter_count = 128;
	auto bn_8_beta = load_weight_new(weight_fname, bn_8_layer_name_1, bn_8_layer_name_2, "beta:0", bn_8_filter_count);
	auto bn_8_gamma = load_weight_new(weight_fname, bn_8_layer_name_1, bn_8_layer_name_2, "gamma:0", bn_8_filter_count);
	auto bn_8_mean = load_weight_new(weight_fname, bn_8_layer_name_1, bn_8_layer_name_2, "moving_mean:0", bn_8_filter_count);
	auto bn_8_variance = load_weight_new(weight_fname, bn_8_layer_name_1, bn_8_layer_name_2, "moving_variance:0", bn_8_filter_count);

	std::vector<float> bn_8_shift_v(bn_8_filter_count, 0.0);
	std::vector<float> bn_8_scale_v(bn_8_filter_count, 0.0);
	std::vector<float> bn_8_power_v(bn_8_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_8_filter_count; ++c_idx)
	{
		bn_8_shift_v[c_idx] = bn_8_beta[c_idx] - (bn_8_gamma[c_idx] * bn_8_mean[c_idx] / sqrtf(bn_8_variance[c_idx] + 0.001));
		bn_8_scale_v[c_idx] = bn_8_gamma[c_idx] / sqrtf(bn_8_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_8_shift_w{ nvinfer1::DataType::kFLOAT, bn_8_shift_v.data(), bn_8_filter_count };
	nvinfer1::Weights bn_8_scale_w{ nvinfer1::DataType::kFLOAT, bn_8_scale_v.data(), bn_8_filter_count };
	nvinfer1::Weights bn_8_power_w{ nvinfer1::DataType::kFLOAT, bn_8_power_v.data(), bn_8_filter_count };
	auto bn_8 = network->addScale(*(conv_8->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_8_shift_w, bn_8_scale_w, bn_8_power_w);

	//  Add Activation
	auto relu_8 = network->addActivation(*(bn_8->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_4 = network->addPooling(*(relu_8->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));

	/* ------------	*/
	/*	Block 9		*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_9_kernel = load_weight_new(weight_fname, "conv2d_9", "conv2d_9_1", "kernel:0", 3 * 3 * 128 * 256);
	auto conv_9_bias = load_weight_new(weight_fname, "conv2d_9", "conv2d_9_1", "bias:0", 256);
	//  Convert to TRT format
	nvinfer1::Weights conv_9_kernel_w{ nvinfer1::DataType::kFLOAT, conv_9_kernel.data(), 3 * 3 * 128 * 256 };
	nvinfer1::Weights conv_9_bias_w{ nvinfer1::DataType::kFLOAT, conv_9_bias.data(), 256 };
	//  Add conv
	auto pad_9 = network->addPadding(*(pool_4->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_9 != nullptr);
	auto conv_9 = network->addConvolution(*(pad_9->getOutput(0)), 256, nvinfer1::DimsHW(3, 3), conv_9_kernel_w, conv_9_bias_w);
	assert(conv_9 != nullptr);
	//  Add BN
	auto bn_9_layer_name_1 = "batch_normalization_9";
	auto bn_9_layer_name_2 = "batch_normalization_9_1";
	auto bn_9_filter_count = 256;
	auto bn_9_beta = load_weight_new(weight_fname, bn_9_layer_name_1, bn_9_layer_name_2, "beta:0", bn_9_filter_count);
	auto bn_9_gamma = load_weight_new(weight_fname, bn_9_layer_name_1, bn_9_layer_name_2, "gamma:0", bn_9_filter_count);
	auto bn_9_mean = load_weight_new(weight_fname, bn_9_layer_name_1, bn_9_layer_name_2, "moving_mean:0", bn_9_filter_count);
	auto bn_9_variance = load_weight_new(weight_fname, bn_9_layer_name_1, bn_9_layer_name_2, "moving_variance:0", bn_9_filter_count);
	std::vector<float> bn_9_shift_v(bn_9_filter_count, 0.0);
	std::vector<float> bn_9_scale_v(bn_9_filter_count, 0.0);
	std::vector<float> bn_9_power_v(bn_9_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_9_filter_count; ++c_idx)
	{
		bn_9_shift_v[c_idx] = bn_9_beta[c_idx] - (bn_9_gamma[c_idx] * bn_9_mean[c_idx] / sqrtf(bn_9_variance[c_idx] + 0.001));
		bn_9_scale_v[c_idx] = bn_9_gamma[c_idx] / sqrtf(bn_9_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_9_shift_w{ nvinfer1::DataType::kFLOAT, bn_9_shift_v.data(), bn_9_filter_count };
	nvinfer1::Weights bn_9_scale_w{ nvinfer1::DataType::kFLOAT, bn_9_scale_v.data(), bn_9_filter_count };
	nvinfer1::Weights bn_9_power_w{ nvinfer1::DataType::kFLOAT, bn_9_power_v.data(), bn_9_filter_count };
	auto bn_9 = network->addScale(*(conv_9->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_9_shift_w, bn_9_scale_w, bn_9_power_w);

	//  Add Activation
	auto relu_9 = network->addActivation(*(bn_9->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 10	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_10_kernel = load_weight_new(weight_fname, "conv2d_10", "conv2d_10_1", "kernel:0", 3 * 3 * 256 * 256);
	auto conv_10_bias = load_weight_new(weight_fname, "conv2d_10", "conv2d_10_1", "bias:0", 256);
	//  Convert to TRT format
	nvinfer1::Weights conv_10_kernel_w{ nvinfer1::DataType::kFLOAT, conv_10_kernel.data(), 3 * 3 * 256 * 256 };
	nvinfer1::Weights conv_10_bias_w{ nvinfer1::DataType::kFLOAT, conv_10_bias.data(), 256 };
	//  Add conv
	auto pad_10 = network->addPadding(*(relu_9->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_10 != nullptr);
	auto conv_10 = network->addConvolution(*(pad_10->getOutput(0)), 256, nvinfer1::DimsHW(3, 3), conv_10_kernel_w, conv_10_bias_w);
	assert(conv_10 != nullptr);
	//  Add BN
	auto bn_10_layer_name_1 = "batch_normalization_10";
	auto bn_10_layer_name_2 = "batch_normalization_10_1";
	auto bn_10_filter_count = 256;
	auto bn_10_beta = load_weight_new(weight_fname, bn_10_layer_name_1, bn_10_layer_name_2, "beta:0", bn_10_filter_count);
	auto bn_10_gamma = load_weight_new(weight_fname, bn_10_layer_name_1, bn_10_layer_name_2, "gamma:0", bn_10_filter_count);
	auto bn_10_mean = load_weight_new(weight_fname, bn_10_layer_name_1, bn_10_layer_name_2, "moving_mean:0", bn_10_filter_count);
	auto bn_10_variance = load_weight_new(weight_fname, bn_10_layer_name_1, bn_10_layer_name_2, "moving_variance:0", bn_10_filter_count);

	std::vector<float> bn_10_shift_v(bn_10_filter_count, 0.0);
	std::vector<float> bn_10_scale_v(bn_10_filter_count, 0.0);
	std::vector<float> bn_10_power_v(bn_10_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_10_filter_count; ++c_idx)
	{
		bn_10_shift_v[c_idx] = bn_10_beta[c_idx] - (bn_10_gamma[c_idx] * bn_10_mean[c_idx] / sqrtf(bn_10_variance[c_idx] + 0.001));
		bn_10_scale_v[c_idx] = bn_10_gamma[c_idx] / sqrtf(bn_10_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_10_shift_w{ nvinfer1::DataType::kFLOAT, bn_10_shift_v.data(), bn_10_filter_count };
	nvinfer1::Weights bn_10_scale_w{ nvinfer1::DataType::kFLOAT, bn_10_scale_v.data(), bn_10_filter_count };
	nvinfer1::Weights bn_10_power_w{ nvinfer1::DataType::kFLOAT, bn_10_power_v.data(), bn_10_filter_count };
	auto bn_10 = network->addScale(*(conv_10->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_10_shift_w, bn_10_scale_w, bn_10_power_w);

	//  Add Activation
	auto relu_10 = network->addActivation(*(bn_10->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------------------------------*/
	/*	1st Up-sampling and Concatenation */
	/* ------------------------------------*/

	auto up_1 = network->addResize(*(relu_10->getOutput(0)));
	std::vector<float> up_scales{ 1, 1, 2, 2 };
	up_1->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_1->setScales(up_scales.data(), 4);

	nvinfer1::ITensor** concat_1_inputs = new nvinfer1::ITensor * [2];
	concat_1_inputs[0] = relu_8->getOutput(0);
	concat_1_inputs[1] = up_1->getOutput(0);
	auto concat_1 = network->addConcatenation(concat_1_inputs, 2);

	/* ------------	*/
	/*	Block 11	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_11_kernel = load_weight_new(weight_fname, "conv2d_11", "conv2d_11_1", "kernel:0", 3 * 3 * 384 * 128);
	auto conv_11_bias = load_weight_new(weight_fname, "conv2d_11", "conv2d_11_1", "bias:0", 128);
	//  Convert to TRT format
	nvinfer1::Weights conv_11_kernel_w{ nvinfer1::DataType::kFLOAT, conv_11_kernel.data(), 3 * 3 * 384 * 128 };
	nvinfer1::Weights conv_11_bias_w{ nvinfer1::DataType::kFLOAT, conv_11_bias.data(), 128 };
	//  Add conv
	auto pad_11 = network->addPadding(*(concat_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_11 != nullptr);
	auto conv_11 = network->addConvolution(*(pad_11->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), conv_11_kernel_w, conv_11_bias_w);
	assert(conv_11 != nullptr);
	//  Add BN
	auto bn_11_layer_name_1 = "batch_normalization_11";
	auto bn_11_layer_name_2 = "batch_normalization_11_1";
	auto bn_11_filter_count = 128;
	auto bn_11_beta = load_weight_new(weight_fname, bn_11_layer_name_1, bn_11_layer_name_2, "beta:0", bn_11_filter_count);
	auto bn_11_gamma = load_weight_new(weight_fname, bn_11_layer_name_1, bn_11_layer_name_2, "gamma:0", bn_11_filter_count);
	auto bn_11_mean = load_weight_new(weight_fname, bn_11_layer_name_1, bn_11_layer_name_2, "moving_mean:0", bn_11_filter_count);
	auto bn_11_variance = load_weight_new(weight_fname, bn_11_layer_name_1, bn_11_layer_name_2, "moving_variance:0", bn_11_filter_count);
	std::vector<float> bn_11_shift_v(bn_11_filter_count, 0.0);
	std::vector<float> bn_11_scale_v(bn_11_filter_count, 0.0);
	std::vector<float> bn_11_power_v(bn_11_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_11_filter_count; ++c_idx)
	{
		bn_11_shift_v[c_idx] = bn_11_beta[c_idx] - (bn_11_gamma[c_idx] * bn_11_mean[c_idx] / sqrtf(bn_11_variance[c_idx] + 0.001));
		bn_11_scale_v[c_idx] = bn_11_gamma[c_idx] / sqrtf(bn_11_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_11_shift_w{ nvinfer1::DataType::kFLOAT, bn_11_shift_v.data(), bn_11_filter_count };
	nvinfer1::Weights bn_11_scale_w{ nvinfer1::DataType::kFLOAT, bn_11_scale_v.data(), bn_11_filter_count };
	nvinfer1::Weights bn_11_power_w{ nvinfer1::DataType::kFLOAT, bn_11_power_v.data(), bn_11_filter_count };
	auto bn_11 = network->addScale(*(conv_11->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_11_shift_w, bn_11_scale_w, bn_11_power_w);

	//  Add Activation
	auto relu_11 = network->addActivation(*(bn_11->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 12	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_12_kernel = load_weight_new(weight_fname, "conv2d_12", "conv2d_12_1", "kernel:0", 3 * 3 * 128 * 128);
	auto conv_12_bias = load_weight_new(weight_fname, "conv2d_12", "conv2d_12_1", "bias:0", 128);
	//  Convert to TRT format
	nvinfer1::Weights conv_12_kernel_w{ nvinfer1::DataType::kFLOAT, conv_12_kernel.data(), 3 * 3 * 128 * 128 };
	nvinfer1::Weights conv_12_bias_w{ nvinfer1::DataType::kFLOAT, conv_12_bias.data(), 128 };
	//  Add conv
	auto pad_12 = network->addPadding(*(relu_11->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_12 != nullptr);
	auto conv_12 = network->addConvolution(*(pad_12->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), conv_12_kernel_w, conv_12_bias_w);
	assert(conv_12 != nullptr);
	//  Add BN
	auto bn_12_layer_name_1 = "batch_normalization_12";
	auto bn_12_layer_name_2 = "batch_normalization_12_1";
	auto bn_12_filter_count = 128;
	auto bn_12_beta = load_weight_new(weight_fname, bn_12_layer_name_1, bn_12_layer_name_2, "beta:0", bn_12_filter_count);
	auto bn_12_gamma = load_weight_new(weight_fname, bn_12_layer_name_1, bn_12_layer_name_2, "gamma:0", bn_12_filter_count);
	auto bn_12_mean = load_weight_new(weight_fname, bn_12_layer_name_1, bn_12_layer_name_2, "moving_mean:0", bn_12_filter_count);
	auto bn_12_variance = load_weight_new(weight_fname, bn_12_layer_name_1, bn_12_layer_name_2, "moving_variance:0", bn_12_filter_count);

	std::vector<float> bn_12_shift_v(bn_12_filter_count, 0.0);
	std::vector<float> bn_12_scale_v(bn_12_filter_count, 0.0);
	std::vector<float> bn_12_power_v(bn_12_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_12_filter_count; ++c_idx)
	{
		bn_12_shift_v[c_idx] = bn_12_beta[c_idx] - (bn_12_gamma[c_idx] * bn_12_mean[c_idx] / sqrtf(bn_12_variance[c_idx] + 0.001));
		bn_12_scale_v[c_idx] = bn_12_gamma[c_idx] / sqrtf(bn_12_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_12_shift_w{ nvinfer1::DataType::kFLOAT, bn_12_shift_v.data(), bn_12_filter_count };
	nvinfer1::Weights bn_12_scale_w{ nvinfer1::DataType::kFLOAT, bn_12_scale_v.data(), bn_12_filter_count };
	nvinfer1::Weights bn_12_power_w{ nvinfer1::DataType::kFLOAT, bn_12_power_v.data(), bn_12_filter_count };
	auto bn_12 = network->addScale(*(conv_12->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_12_shift_w, bn_12_scale_w, bn_12_power_w);

	//  Add Activation
	auto relu_12 = network->addActivation(*(bn_12->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------------------------------*/
	/*	2nd Up-sampling and Concatenation */
	/* ------------------------------------*/

	auto up_2 = network->addResize(*(relu_12->getOutput(0)));
	up_2->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_2->setScales(up_scales.data(), 4);

	nvinfer1::ITensor** concat_2_inputs = new nvinfer1::ITensor * [2];
	concat_2_inputs[0] = relu_6->getOutput(0);
	concat_2_inputs[1] = up_2->getOutput(0);
	auto concat_2 = network->addConcatenation(concat_2_inputs, 2);

	/* ------------	*/
	/*	Block 13	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_13_kernel = load_weight_new(weight_fname, "conv2d_13", "conv2d_13_1", "kernel:0", 3 * 3 * 192 * 64);
	auto conv_13_bias = load_weight_new(weight_fname, "conv2d_13", "conv2d_13_1", "bias:0", 64);
	//  Convert to TRT format
	nvinfer1::Weights conv_13_kernel_w{ nvinfer1::DataType::kFLOAT, conv_13_kernel.data(), 3 * 3 * 192 * 64 };
	nvinfer1::Weights conv_13_bias_w{ nvinfer1::DataType::kFLOAT, conv_13_bias.data(), 64 };
	//  Add conv
	auto pad_13 = network->addPadding(*(concat_2->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_13 != nullptr);
	auto conv_13 = network->addConvolution(*(pad_13->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), conv_13_kernel_w, conv_13_bias_w);
	assert(conv_13 != nullptr);
	//  Add BN
	auto bn_13_layer_name_1 = "batch_normalization_13";
	auto bn_13_layer_name_2 = "batch_normalization_13_1";
	auto bn_13_filter_count = 64;
	auto bn_13_beta = load_weight_new(weight_fname, bn_13_layer_name_1, bn_13_layer_name_2, "beta:0", bn_13_filter_count);
	auto bn_13_gamma = load_weight_new(weight_fname, bn_13_layer_name_1, bn_13_layer_name_2, "gamma:0", bn_13_filter_count);
	auto bn_13_mean = load_weight_new(weight_fname, bn_13_layer_name_1, bn_13_layer_name_2, "moving_mean:0", bn_13_filter_count);
	auto bn_13_variance = load_weight_new(weight_fname, bn_13_layer_name_1, bn_13_layer_name_2, "moving_variance:0", bn_13_filter_count);
	std::vector<float> bn_13_shift_v(bn_13_filter_count, 0.0);
	std::vector<float> bn_13_scale_v(bn_13_filter_count, 0.0);
	std::vector<float> bn_13_power_v(bn_13_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_13_filter_count; ++c_idx)
	{
		bn_13_shift_v[c_idx] = bn_13_beta[c_idx] - (bn_13_gamma[c_idx] * bn_13_mean[c_idx] / sqrtf(bn_13_variance[c_idx] + 0.001));
		bn_13_scale_v[c_idx] = bn_13_gamma[c_idx] / sqrtf(bn_13_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_13_shift_w{ nvinfer1::DataType::kFLOAT, bn_13_shift_v.data(), bn_13_filter_count };
	nvinfer1::Weights bn_13_scale_w{ nvinfer1::DataType::kFLOAT, bn_13_scale_v.data(), bn_13_filter_count };
	nvinfer1::Weights bn_13_power_w{ nvinfer1::DataType::kFLOAT, bn_13_power_v.data(), bn_13_filter_count };
	auto bn_13 = network->addScale(*(conv_13->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_13_shift_w, bn_13_scale_w, bn_13_power_w);

	//  Add Activation
	auto relu_13 = network->addActivation(*(bn_13->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 14	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_14_kernel = load_weight_new(weight_fname, "conv2d_14", "conv2d_14_1", "kernel:0", 3 * 3 * 64 * 64);
	auto conv_14_bias = load_weight_new(weight_fname, "conv2d_14", "conv2d_14_1", "bias:0", 64);
	//  Convert to TRT format
	nvinfer1::Weights conv_14_kernel_w{ nvinfer1::DataType::kFLOAT, conv_14_kernel.data(), 3 * 3 * 64 * 64 };
	nvinfer1::Weights conv_14_bias_w{ nvinfer1::DataType::kFLOAT, conv_14_bias.data(), 64 };
	//  Add conv
	auto pad_14 = network->addPadding(*(relu_13->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_14 != nullptr);
	auto conv_14 = network->addConvolution(*(pad_14->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), conv_14_kernel_w, conv_14_bias_w);
	assert(conv_14 != nullptr);
	//  Add BN
	auto bn_14_layer_name_1 = "batch_normalization_14";
	auto bn_14_layer_name_2 = "batch_normalization_14_1";
	auto bn_14_filter_count = 64;
	auto bn_14_beta = load_weight_new(weight_fname, bn_14_layer_name_1, bn_14_layer_name_2, "beta:0", bn_14_filter_count);
	auto bn_14_gamma = load_weight_new(weight_fname, bn_14_layer_name_1, bn_14_layer_name_2, "gamma:0", bn_14_filter_count);
	auto bn_14_mean = load_weight_new(weight_fname, bn_14_layer_name_1, bn_14_layer_name_2, "moving_mean:0", bn_14_filter_count);
	auto bn_14_variance = load_weight_new(weight_fname, bn_14_layer_name_1, bn_14_layer_name_2, "moving_variance:0", bn_14_filter_count);

	std::vector<float> bn_14_shift_v(bn_14_filter_count, 0.0);
	std::vector<float> bn_14_scale_v(bn_14_filter_count, 0.0);
	std::vector<float> bn_14_power_v(bn_14_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_14_filter_count; ++c_idx)
	{
		bn_14_shift_v[c_idx] = bn_14_beta[c_idx] - (bn_14_gamma[c_idx] * bn_14_mean[c_idx] / sqrtf(bn_14_variance[c_idx] + 0.001));
		bn_14_scale_v[c_idx] = bn_14_gamma[c_idx] / sqrtf(bn_14_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_14_shift_w{ nvinfer1::DataType::kFLOAT, bn_14_shift_v.data(), bn_14_filter_count };
	nvinfer1::Weights bn_14_scale_w{ nvinfer1::DataType::kFLOAT, bn_14_scale_v.data(), bn_14_filter_count };
	nvinfer1::Weights bn_14_power_w{ nvinfer1::DataType::kFLOAT, bn_14_power_v.data(), bn_14_filter_count };
	auto bn_14 = network->addScale(*(conv_14->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_14_shift_w, bn_14_scale_w, bn_14_power_w);

	//  Add Activation
	auto relu_14 = network->addActivation(*(bn_14->getOutput(0)), nvinfer1::ActivationType::kRELU);


	/* ------------------------------------*/
	/*	3rd Up-sampling and Concatenation */
	/* ------------------------------------*/

	auto up_3 = network->addResize(*(relu_14->getOutput(0)));
	up_3->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_3->setScales(up_scales.data(), 4);

	nvinfer1::ITensor** concat_3_inputs = new nvinfer1::ITensor * [2];
	concat_3_inputs[0] = relu_4->getOutput(0);
	concat_3_inputs[1] = up_3->getOutput(0);
	auto concat_3 = network->addConcatenation(concat_3_inputs, 2);

	/* ------------	*/
	/*	Block 15	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_15_kernel = load_weight_new(weight_fname, "conv2d_15", "conv2d_15_1", "kernel:0", 3 * 3 * 96 * 32);
	auto conv_15_bias = load_weight_new(weight_fname, "conv2d_15", "conv2d_15_1", "bias:0", 32);
	//  Convert to TRT format
	nvinfer1::Weights conv_15_kernel_w{ nvinfer1::DataType::kFLOAT, conv_15_kernel.data(), 3 * 3 * 96 * 32 };
	nvinfer1::Weights conv_15_bias_w{ nvinfer1::DataType::kFLOAT, conv_15_bias.data(), 32 };
	//  Add conv
	auto pad_15 = network->addPadding(*(concat_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_15 != nullptr);
	auto conv_15 = network->addConvolution(*(pad_15->getOutput(0)), 32, nvinfer1::DimsHW(3, 3), conv_15_kernel_w, conv_15_bias_w);
	assert(conv_15 != nullptr);
	//  Add BN
	auto bn_15_layer_name_1 = "batch_normalization_15";
	auto bn_15_layer_name_2 = "batch_normalization_15_1";
	auto bn_15_filter_count = 32;
	auto bn_15_beta = load_weight_new(weight_fname, bn_15_layer_name_1, bn_15_layer_name_2, "beta:0", bn_15_filter_count);
	auto bn_15_gamma = load_weight_new(weight_fname, bn_15_layer_name_1, bn_15_layer_name_2, "gamma:0", bn_15_filter_count);
	auto bn_15_mean = load_weight_new(weight_fname, bn_15_layer_name_1, bn_15_layer_name_2, "moving_mean:0", bn_15_filter_count);
	auto bn_15_variance = load_weight_new(weight_fname, bn_15_layer_name_1, bn_15_layer_name_2, "moving_variance:0", bn_15_filter_count);
	std::vector<float> bn_15_shift_v(bn_15_filter_count, 0.0);
	std::vector<float> bn_15_scale_v(bn_15_filter_count, 0.0);
	std::vector<float> bn_15_power_v(bn_15_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_15_filter_count; ++c_idx)
	{
		bn_15_shift_v[c_idx] = bn_15_beta[c_idx] - (bn_15_gamma[c_idx] * bn_15_mean[c_idx] / sqrtf(bn_15_variance[c_idx] + 0.001));
		bn_15_scale_v[c_idx] = bn_15_gamma[c_idx] / sqrtf(bn_15_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_15_shift_w{ nvinfer1::DataType::kFLOAT, bn_15_shift_v.data(), bn_15_filter_count };
	nvinfer1::Weights bn_15_scale_w{ nvinfer1::DataType::kFLOAT, bn_15_scale_v.data(), bn_15_filter_count };
	nvinfer1::Weights bn_15_power_w{ nvinfer1::DataType::kFLOAT, bn_15_power_v.data(), bn_15_filter_count };
	auto bn_15 = network->addScale(*(conv_15->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_15_shift_w, bn_15_scale_w, bn_15_power_w);

	//  Add Activation
	auto relu_15 = network->addActivation(*(bn_15->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 16	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_16_kernel = load_weight_new(weight_fname, "conv2d_16", "conv2d_16_1", "kernel:0", 3 * 3 * 32 * 32);
	auto conv_16_bias = load_weight_new(weight_fname, "conv2d_16", "conv2d_16_1", "bias:0", 32);
	//  Convert to TRT format
	nvinfer1::Weights conv_16_kernel_w{ nvinfer1::DataType::kFLOAT, conv_16_kernel.data(), 3 * 3 * 32 * 32 };
	nvinfer1::Weights conv_16_bias_w{ nvinfer1::DataType::kFLOAT, conv_16_bias.data(), 32 };
	//  Add conv
	auto pad_16 = network->addPadding(*(relu_15->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_16 != nullptr);
	auto conv_16 = network->addConvolution(*(pad_16->getOutput(0)), 32, nvinfer1::DimsHW(3, 3), conv_16_kernel_w, conv_16_bias_w);
	assert(conv_16 != nullptr);
	//  Add BN
	auto bn_16_layer_name_1 = "batch_normalization_16";
	auto bn_16_layer_name_2 = "batch_normalization_16_1";
	auto bn_16_filter_count = 32;
	auto bn_16_beta = load_weight_new(weight_fname, bn_16_layer_name_1, bn_16_layer_name_2, "beta:0", bn_16_filter_count);
	auto bn_16_gamma = load_weight_new(weight_fname, bn_16_layer_name_1, bn_16_layer_name_2, "gamma:0", bn_16_filter_count);
	auto bn_16_mean = load_weight_new(weight_fname, bn_16_layer_name_1, bn_16_layer_name_2, "moving_mean:0", bn_16_filter_count);
	auto bn_16_variance = load_weight_new(weight_fname, bn_16_layer_name_1, bn_16_layer_name_2, "moving_variance:0", bn_16_filter_count);

	std::vector<float> bn_16_shift_v(bn_16_filter_count, 0.0);
	std::vector<float> bn_16_scale_v(bn_16_filter_count, 0.0);
	std::vector<float> bn_16_power_v(bn_16_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_16_filter_count; ++c_idx)
	{
		bn_16_shift_v[c_idx] = bn_16_beta[c_idx] - (bn_16_gamma[c_idx] * bn_16_mean[c_idx] / sqrtf(bn_16_variance[c_idx] + 0.001));
		bn_16_scale_v[c_idx] = bn_16_gamma[c_idx] / sqrtf(bn_16_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_16_shift_w{ nvinfer1::DataType::kFLOAT, bn_16_shift_v.data(), bn_16_filter_count };
	nvinfer1::Weights bn_16_scale_w{ nvinfer1::DataType::kFLOAT, bn_16_scale_v.data(), bn_16_filter_count };
	nvinfer1::Weights bn_16_power_w{ nvinfer1::DataType::kFLOAT, bn_16_power_v.data(), bn_16_filter_count };
	auto bn_16 = network->addScale(*(conv_16->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_16_shift_w, bn_16_scale_w, bn_16_power_w);

	//  Add Activation
	auto relu_16 = network->addActivation(*(bn_16->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------------------------------*/
	/*	4th Up-sampling and Concatenation */
	/* ------------------------------------*/

	auto up_4 = network->addResize(*(relu_16->getOutput(0)));
	up_4->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_4->setScales(up_scales.data(), 4);

	nvinfer1::ITensor** concat_4_inputs = new nvinfer1::ITensor * [2];
	concat_4_inputs[0] = relu_2->getOutput(0);
	concat_4_inputs[1] = up_4->getOutput(0);
	auto concat_4 = network->addConcatenation(concat_4_inputs, 2);

	/* ------------	*/
	/*	Block 17	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_17_kernel = load_weight_new(weight_fname, "conv2d_17", "conv2d_17_1", "kernel:0", 3 * 3 * 48 * 16);
	auto conv_17_bias = load_weight_new(weight_fname, "conv2d_17", "conv2d_17_1", "bias:0", 16);
	//  Convert to TRT format
	nvinfer1::Weights conv_17_kernel_w{ nvinfer1::DataType::kFLOAT, conv_17_kernel.data(), 3 * 3 * 48 * 16 };
	nvinfer1::Weights conv_17_bias_w{ nvinfer1::DataType::kFLOAT, conv_17_bias.data(), 16 };
	//  Add conv
	auto pad_17 = network->addPadding(*(concat_4->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_17 != nullptr);
	auto conv_17 = network->addConvolution(*(pad_17->getOutput(0)), 16, nvinfer1::DimsHW(3, 3), conv_17_kernel_w, conv_17_bias_w);
	assert(conv_17 != nullptr);
	//  Add BN
	auto bn_17_layer_name_1 = "batch_normalization_17";
	auto bn_17_layer_name_2 = "batch_normalization_17_1";
	auto bn_17_filter_count = 16;
	auto bn_17_beta = load_weight_new(weight_fname, bn_17_layer_name_1, bn_17_layer_name_2, "beta:0", bn_17_filter_count);
	auto bn_17_gamma = load_weight_new(weight_fname, bn_17_layer_name_1, bn_17_layer_name_2, "gamma:0", bn_17_filter_count);
	auto bn_17_mean = load_weight_new(weight_fname, bn_17_layer_name_1, bn_17_layer_name_2, "moving_mean:0", bn_17_filter_count);
	auto bn_17_variance = load_weight_new(weight_fname, bn_17_layer_name_1, bn_17_layer_name_2, "moving_variance:0", bn_17_filter_count);
	std::vector<float> bn_17_shift_v(bn_17_filter_count, 0.0);
	std::vector<float> bn_17_scale_v(bn_17_filter_count, 0.0);
	std::vector<float> bn_17_power_v(bn_17_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_17_filter_count; ++c_idx)
	{
		bn_17_shift_v[c_idx] = bn_17_beta[c_idx] - (bn_17_gamma[c_idx] * bn_17_mean[c_idx] / sqrtf(bn_17_variance[c_idx] + 0.001));
		bn_17_scale_v[c_idx] = bn_17_gamma[c_idx] / sqrtf(bn_17_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_17_shift_w{ nvinfer1::DataType::kFLOAT, bn_17_shift_v.data(), bn_17_filter_count };
	nvinfer1::Weights bn_17_scale_w{ nvinfer1::DataType::kFLOAT, bn_17_scale_v.data(), bn_17_filter_count };
	nvinfer1::Weights bn_17_power_w{ nvinfer1::DataType::kFLOAT, bn_17_power_v.data(), bn_17_filter_count };
	auto bn_17 = network->addScale(*(conv_17->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_17_shift_w, bn_17_scale_w, bn_17_power_w);

	//  Add Activation
	auto relu_17 = network->addActivation(*(bn_17->getOutput(0)), nvinfer1::ActivationType::kRELU);


	/* ------------	*/
	/*	Block 18	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_18_kernel = load_weight_new(weight_fname, "conv2d_18", "conv2d_18_1", "kernel:0", 3 * 3 * 16 * 16);
	auto conv_18_bias = load_weight_new(weight_fname, "conv2d_18", "conv2d_18_1", "bias:0", 16);
	//  Convert to TRT format
	nvinfer1::Weights conv_18_kernel_w{ nvinfer1::DataType::kFLOAT, conv_18_kernel.data(), 3 * 3 * 16 * 16 };
	nvinfer1::Weights conv_18_bias_w{ nvinfer1::DataType::kFLOAT, conv_18_bias.data(), 16 };
	//  Add conv
	auto pad_18 = network->addPadding(*(relu_17->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_18 != nullptr);
	auto conv_18 = network->addConvolution(*(pad_18->getOutput(0)), 16, nvinfer1::DimsHW(3, 3), conv_18_kernel_w, conv_18_bias_w);
	assert(conv_18 != nullptr);
	//  Add BN
	auto bn_18_layer_name_1 = "batch_normalization_18";
	auto bn_18_layer_name_2 = "batch_normalization_18_1";
	auto bn_18_filter_count = 16;
	auto bn_18_beta = load_weight_new(weight_fname, bn_18_layer_name_1, bn_18_layer_name_2, "beta:0", bn_18_filter_count);
	auto bn_18_gamma = load_weight_new(weight_fname, bn_18_layer_name_1, bn_18_layer_name_2, "gamma:0", bn_18_filter_count);
	auto bn_18_mean = load_weight_new(weight_fname, bn_18_layer_name_1, bn_18_layer_name_2, "moving_mean:0", bn_18_filter_count);
	auto bn_18_variance = load_weight_new(weight_fname, bn_18_layer_name_1, bn_18_layer_name_2, "moving_variance:0", bn_18_filter_count);

	std::vector<float> bn_18_shift_v(bn_18_filter_count, 0.0);
	std::vector<float> bn_18_scale_v(bn_18_filter_count, 0.0);
	std::vector<float> bn_18_power_v(bn_18_filter_count, 1.0);
	for (auto c_idx = 0; c_idx < bn_18_filter_count; ++c_idx)
	{
		bn_18_shift_v[c_idx] = bn_18_beta[c_idx] - (bn_18_gamma[c_idx] * bn_18_mean[c_idx] / sqrtf(bn_18_variance[c_idx] + 0.001));
		bn_18_scale_v[c_idx] = bn_18_gamma[c_idx] / sqrtf(bn_18_variance[c_idx] + 0.001);
	}
	nvinfer1::Weights bn_18_shift_w{ nvinfer1::DataType::kFLOAT, bn_18_shift_v.data(), bn_18_filter_count };
	nvinfer1::Weights bn_18_scale_w{ nvinfer1::DataType::kFLOAT, bn_18_scale_v.data(), bn_18_filter_count };
	nvinfer1::Weights bn_18_power_w{ nvinfer1::DataType::kFLOAT, bn_18_power_v.data(), bn_18_filter_count };
	auto bn_18 = network->addScale(*(conv_18->getOutput(0)), nvinfer1::ScaleMode::kCHANNEL, bn_18_shift_w, bn_18_scale_w, bn_18_power_w);

	//  Add Activation
	auto relu_18 = network->addActivation(*(bn_18->getOutput(0)), nvinfer1::ActivationType::kRELU);

	/* ------------	*/
	/*	Block 19	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_19_kernel = load_weight_new(weight_fname, "conv2d_19", "conv2d_19_1", "kernel:0", 3 * 3 * 16 * 1);
	auto conv_19_bias = load_weight_new(weight_fname, "conv2d_19", "conv2d_19_1", "bias:0", 1);
	//  Convert to TRT format
	nvinfer1::Weights conv_19_kernel_w{ nvinfer1::DataType::kFLOAT, conv_19_kernel.data(), 3 * 3 * 16 * 1 };
	nvinfer1::Weights conv_19_bias_w{ nvinfer1::DataType::kFLOAT, conv_19_bias.data(), 1 };
	//  Add conv
	auto pad_19 = network->addPadding(*(relu_18->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	assert(pad_19 != nullptr);
	auto conv_19 = network->addConvolution(*(pad_19->getOutput(0)), 1, nvinfer1::DimsHW(3, 3), conv_19_kernel_w, conv_19_bias_w);
	assert(conv_19 != nullptr);

	/* -----------------	*/
	/*	Adding with Input	*/
	/* -------------------- */
	auto add_1 = network->addElementWise(*(scaled_input->getOutput(0)), *(conv_19->getOutput(0)), nvinfer1::ElementWiseOperation::kSUM);
	assert(add_1 != nullptr);

	/* ------------	*/
	/*	Block 20	*/
	/* ------------ */
	//  Load the weights form hdf5
	auto conv_20_kernel = load_weight_new(weight_fname, "conv2d_20", "conv2d_20_1", "kernel:0", 1 * 1 * 1 * 1);
	auto conv_20_bias = load_weight_new(weight_fname, "conv2d_20", "conv2d_20_1", "bias:0", 1);
	//  Convert to TRT format
	nvinfer1::Weights conv_20_kernel_w{ nvinfer1::DataType::kFLOAT, conv_20_kernel.data(), 1 * 1 * 1 * 1 };
	nvinfer1::Weights conv_20_bias_w{ nvinfer1::DataType::kFLOAT, conv_20_bias.data(), 1 };
	//  Add conv
	auto conv_20 = network->addConvolution(*(add_1->getOutput(0)), 1, nvinfer1::DimsHW(1, 1), conv_20_kernel_w, conv_20_bias_w);
	assert(conv_20 != nullptr);
	auto out_layer = network->addActivation(*(conv_20->getOutput(0)), nvinfer1::ActivationType::kSIGMOID);

	//  Set the output
	network->markOutput(*(out_layer->getOutput(0)));


	// Build engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	assert(engine != nullptr);

	//  Following sampleMNISTAPI, network is destroyed and host memory for weight is released
	network->destroy();
	return engine;
}

#endif

