#pragma once
#ifndef UNET_ENGINE_V3_H
#define UNET_ENGINE_V3_H

#include <NvInfer.h>
#include <cassert>
#include "h5file.hpp"
#include "ml_shared.h"
#include "qli_runtime_error.h"
#include <iostream>
#include <unordered_map>
#include <string>

nvinfer1::ICudaEngine* create_unet_engine_v3(nvinfer1::IBuilder* builder, const char* weight_fname, int in_c, int in_h, int in_w, float x_min, float x_max)
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

	std::cout << "Loading model weights from " << weight_fname << std::endl;

	// Load all the weights
	/* provided by Python code*/
	std::unordered_map<std::string, std::vector<int>> conv_dict{
		{"conv2d", {3, 1, 16}},
		{"conv2d_1", {3, 16, 16}},
		{"conv2d_10", {3, 128, 128}},
		{"conv2d_11", {1, 64, 128}},
		{"conv2d_12", {3, 128, 256}},
		{"conv2d_13", {3, 256, 256}},
		{"conv2d_14", {1, 128, 256}},
		{"conv2d_15", {3, 256, 256}},
		{"conv2d_16", {3, 256, 256}},
		{"conv2d_17", {1, 256, 256}},
		{"conv2d_18", {3, 384, 128}},
		{"conv2d_19", {3, 128, 128}},
		{"conv2d_2", {1, 1, 16}},
		{"conv2d_20", {1, 384, 128}},
		{"conv2d_21", {3, 192, 64}},
		{"conv2d_22", {3, 64, 64}},
		{"conv2d_23", {1, 192, 64}},
		{"conv2d_24", {3, 96, 32}},
		{"conv2d_25", {3, 32, 32}},
		{"conv2d_26", {1, 96, 32}},
		{"conv2d_27", {3, 48, 16}},
		{"conv2d_28", {3, 16, 16}},
		{"conv2d_29", {1, 48, 16}},
		{"conv2d_3", {3, 16, 32}},
		{"conv2d_30", {3, 16, 1}},
		{"conv2d_31", {1, 1, 1}},
		{"conv2d_4", {3, 32, 32}},
		{"conv2d_5", {1, 16, 32}},
		{"conv2d_6", {3, 32, 64}},
		{"conv2d_7", {3, 64, 64}},
		{"conv2d_8", {1, 32, 64}},
		{"conv2d_9", {3, 64, 128}}
	};
	std::unordered_map<std::string, int> bn_dict{
		{"batch_normalization", {16}},
		{"batch_normalization_1", {16}},
		{"batch_normalization_10", {128}},
		{"batch_normalization_11", {128}},
		{"batch_normalization_12", {256}},
		{"batch_normalization_13", {256}},
		{"batch_normalization_14", {256}},
		{"batch_normalization_15", {256}},
		{"batch_normalization_16", {256}},
		{"batch_normalization_17", {256}},
		{"batch_normalization_18", {128}},
		{"batch_normalization_19", {128}},
		{"batch_normalization_2", {16}},
		{"batch_normalization_20", {128}},
		{"batch_normalization_21", {64}},
		{"batch_normalization_22", {64}},
		{"batch_normalization_23", {64}},
		{"batch_normalization_24", {32}},
		{"batch_normalization_25", {32}},
		{"batch_normalization_26", {32}},
		{"batch_normalization_27", {16}},
		{"batch_normalization_28", {16}},
		{"batch_normalization_29", {16}},
		{"batch_normalization_3", {32}},
		{"batch_normalization_4", {32}},
		{"batch_normalization_5", {32}},
		{"batch_normalization_6", {64}},
		{"batch_normalization_7", {64}},
		{"batch_normalization_8", {64}},
		{"batch_normalization_9", {128}}
	};
	/* fill in */
	std::unordered_map<std::string, std::vector<float>> weight_store;
	for (auto& conv_name : conv_dict)
	{
		auto layer_name = conv_name.first;
		auto sizes = conv_name.second;
		assert(layer_name.find("conv") != std::string::npos);
		auto kernel_size = sizes.at(0), in_filter = sizes.at(1), out_filter = sizes.at(2);
		const auto conv_size = kernel_size * kernel_size * in_filter * out_filter;
		const auto bias_size = out_filter;
		auto kernel_vector = load_weight_new(weight_fname, layer_name.c_str(), layer_name.c_str(), "kernel:0", conv_size);
		auto bias_vector = load_weight_new(weight_fname, layer_name.c_str(), layer_name.c_str(), "bias:0", bias_size);
		auto kernel_trt = nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, kernel_vector.data(), conv_size };
		auto bias_trt = nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, bias_vector.data(), out_filter };
		weight_store[layer_name + "_kernel"] = kernel_vector;
		weight_store[layer_name + "_bias"] = bias_vector;
	}
	for (auto& bn_name : bn_dict)
	{
		auto layer_name = bn_name.first;
		const int out_filter = bn_name.second;
		assert(layer_name.find("batch") != std::string::npos);
		auto beta_vector = load_weight_new(weight_fname, layer_name.c_str(), layer_name.c_str(), "beta:0", out_filter);
		auto gamma_vector = load_weight_new(weight_fname, layer_name.c_str(), layer_name.c_str(), "gamma:0", out_filter);
		auto mean_vector = load_weight_new(weight_fname, layer_name.c_str(), layer_name.c_str(), "moving_mean:0", out_filter);
		auto variance_vector = load_weight_new(weight_fname, layer_name.c_str(), layer_name.c_str(), "moving_variance:0", out_filter);
		std::vector<float> shift_weight(out_filter, 0.0);
		std::vector<float> scale_weight(out_filter, 0.0);
		std::vector<float> power_weight(out_filter, 1.0);
		const auto epsilon = 0.001;
		for (auto c_idx = 0; c_idx < out_filter; ++c_idx)
		{
			auto denominator = sqrtf(variance_vector[c_idx] + epsilon);
			shift_weight[c_idx] = beta_vector[c_idx] - (gamma_vector[c_idx] * mean_vector[c_idx] / denominator);
			scale_weight[c_idx] = gamma_vector[c_idx] / denominator;
		}
		auto shift_trt = nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, shift_weight.data(), out_filter };
		auto scale_trt = nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, scale_weight.data(), out_filter };
		auto power_trt = nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, power_weight.data(), out_filter };
		weight_store[layer_name + "_shift"] = shift_weight;
		weight_store[layer_name + "_scale"] = scale_weight;
		weight_store[layer_name + "_power"] = power_weight;
	}
	std::cout << "All weights read from the h5 file !" << std::endl;

	/*
	*	Scale the input
	*/

	std::vector<float> shift_value{ -x_min / (x_max - x_min) };
	nvinfer1::Weights input_shift{ nvinfer1::DataType::kFLOAT, shift_value.data(), 1 };
	std::vector<float> scale_value{ 1 / (x_max - x_min) };
	nvinfer1::Weights input_scale{ nvinfer1::DataType::kFLOAT, scale_value.data(), 1 };
	std::vector<float> power_value{ 1 };
	nvinfer1::Weights input_power{ nvinfer1::DataType::kFLOAT, power_value.data(), 1 };
	auto scaled_input = network->addScale(*in_tensor, nvinfer1::ScaleMode::kUNIFORM, input_shift, input_scale, input_power);
	assert(scaled_input != nullptr); // (h,w) = (1760, 1776)

	/*
	*	First CBRA Block
	*/

	auto conv2d = network->addConvolutionNd(
		*scaled_input->getOutput(0), conv_dict["conv2d"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d"].at(0), conv_dict["conv2d"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_kernel"].data(), conv_dict["conv2d"].at(0) * conv_dict["conv2d"].at(0) * conv_dict["conv2d"].at(1) * conv_dict["conv2d"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_bias"].data(), conv_dict["conv2d"].at(2) }
	);
	assert(conv2d);
	conv2d->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto bn = network->addScaleNd(
		*conv2d->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_shift"].data(), bn_dict["batch_normalization"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_scale"].data(), bn_dict["batch_normalization"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_power"].data(), bn_dict["batch_normalization"] },
		1
	);
	assert(bn);

	auto activation = network->addActivation(
		*bn->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation);

	auto conv2d_1 = network->addConvolutionNd(
		*activation->getOutput(0), conv_dict["conv2d_1"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_1"].at(0), conv_dict["conv2d_1"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_1_kernel"].data(), conv_dict["conv2d_1"].at(0) * conv_dict["conv2d_1"].at(0) * conv_dict["conv2d_1"].at(1) * conv_dict["conv2d_1"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_1_bias"].data(), conv_dict["conv2d_1"].at(2) }
	);
	assert(conv2d_1);
	conv2d_1->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_1 = network->addScaleNd(
		*conv2d_1->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_1_shift"].data(), bn_dict["batch_normalization_1"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_1_scale"].data(), bn_dict["batch_normalization_1"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_1_power"].data(), bn_dict["batch_normalization_1"] },
		1
	);
	assert(batch_normalization_1);

	auto conv2d_2 = network->addConvolutionNd(
		*scaled_input->getOutput(0), conv_dict["conv2d_2"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_2"].at(0), conv_dict["conv2d_2"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_2_kernel"].data(), conv_dict["conv2d_2"].at(0) * conv_dict["conv2d_2"].at(0) * conv_dict["conv2d_2"].at(1) * conv_dict["conv2d_2"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_2_bias"].data(), conv_dict["conv2d_2"].at(2) }
	);
	assert(conv2d_2);
	conv2d_2->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_2 = network->addScaleNd(
		*conv2d_2->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_2_shift"].data(), bn_dict["batch_normalization_2"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_2_scale"].data(), bn_dict["batch_normalization_2"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_2_power"].data(), bn_dict["batch_normalization_2"] },
		1
	);
	assert(batch_normalization_2);

	auto add = network->addElementWise(
		*batch_normalization_1->getOutput(0),
		*batch_normalization_2->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add);

	auto activation_1 = network->addActivation(*add->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_1);

	auto max_pooling2d = network->addPoolingNd(
		*activation_1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	assert(max_pooling2d);
	/* End of First CBRA Block*/


	/*
	*	Second CBRA Block
	*/

	auto conv2d_3 = network->addConvolutionNd(
		*max_pooling2d->getOutput(0), conv_dict["conv2d_3"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_3"].at(0), conv_dict["conv2d_3"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_3_kernel"].data(), conv_dict["conv2d_3"].at(0) * conv_dict["conv2d_3"].at(0) * conv_dict["conv2d_3"].at(1) * conv_dict["conv2d_3"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_3_bias"].data(), conv_dict["conv2d_3"].at(2) }
	);
	assert(conv2d_3);
	conv2d_3->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_3 = network->addScaleNd(
		*conv2d_3->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_3_shift"].data(), bn_dict["batch_normalization_3"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_3_scale"].data(), bn_dict["batch_normalization_3"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_3_power"].data(), bn_dict["batch_normalization_3"] },
		1
	);
	assert(batch_normalization_3);

	auto activation_2 = network->addActivation(
		*batch_normalization_3->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_2);

	auto conv2d_4 = network->addConvolutionNd(
		*activation_2->getOutput(0), conv_dict["conv2d_4"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_4"].at(0), conv_dict["conv2d_4"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_4_kernel"].data(), conv_dict["conv2d_4"].at(0) * conv_dict["conv2d_4"].at(0) * conv_dict["conv2d_4"].at(1) * conv_dict["conv2d_4"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_4_bias"].data(), conv_dict["conv2d_4"].at(2) }
	);
	assert(conv2d_4);
	conv2d_4->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_4 = network->addScaleNd(
		*conv2d_4->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_4_shift"].data(), bn_dict["batch_normalization_4"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_4_scale"].data(), bn_dict["batch_normalization_4"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_4_power"].data(), bn_dict["batch_normalization_4"] },
		1
	);
	assert(batch_normalization_4);

	auto conv2d_5 = network->addConvolutionNd(
		*max_pooling2d->getOutput(0), conv_dict["conv2d_5"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_5"].at(0), conv_dict["conv2d_5"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_5_kernel"].data(), conv_dict["conv2d_5"].at(0) * conv_dict["conv2d_5"].at(0) * conv_dict["conv2d_5"].at(1) * conv_dict["conv2d_5"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_5_bias"].data(), conv_dict["conv2d_5"].at(2) }
	);
	assert(conv2d_5);
	conv2d_5->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_5 = network->addScaleNd(
		*conv2d_5->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_5_shift"].data(), bn_dict["batch_normalization_5"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_5_scale"].data(), bn_dict["batch_normalization_5"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_5_power"].data(), bn_dict["batch_normalization_5"] },
		1
	);
	assert(batch_normalization_5);

	auto add_1 = network->addElementWise(
		*batch_normalization_4->getOutput(0),
		*batch_normalization_5->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_1);

	auto activation_3 = network->addActivation(*add_1->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_3);

	auto max_pooling2d_1 = network->addPoolingNd(
		*activation_3->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	assert(max_pooling2d_1);
	/* End of Second CBRA Block*/

	/*
	*	Third CBRA Block
	*/
	auto conv2d_6 = network->addConvolutionNd(
		*max_pooling2d_1->getOutput(0), conv_dict["conv2d_6"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_6"].at(0), conv_dict["conv2d_6"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_6_kernel"].data(), conv_dict["conv2d_6"].at(0) * conv_dict["conv2d_6"].at(0) * conv_dict["conv2d_6"].at(1) * conv_dict["conv2d_6"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_6_bias"].data(), conv_dict["conv2d_6"].at(2) }
	);
	assert(conv2d_6);
	conv2d_6->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_6 = network->addScaleNd(
		*conv2d_6->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_6_shift"].data(), bn_dict["batch_normalization_6"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_6_scale"].data(), bn_dict["batch_normalization_6"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_6_power"].data(), bn_dict["batch_normalization_6"] },
		1
	);
	assert(batch_normalization_6);

	auto activation_4 = network->addActivation(
		*batch_normalization_6->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_4);

	auto conv2d_7 = network->addConvolutionNd(
		*activation_4->getOutput(0), conv_dict["conv2d_7"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_7"].at(0), conv_dict["conv2d_7"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_7_kernel"].data(), conv_dict["conv2d_7"].at(0) * conv_dict["conv2d_7"].at(0) * conv_dict["conv2d_7"].at(1) * conv_dict["conv2d_7"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_7_bias"].data(), conv_dict["conv2d_7"].at(2) }
	);
	assert(conv2d_7);
	conv2d_7->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_7 = network->addScaleNd(
		*conv2d_7->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_7_shift"].data(), bn_dict["batch_normalization_7"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_7_scale"].data(), bn_dict["batch_normalization_7"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_7_power"].data(), bn_dict["batch_normalization_7"] },
		1
	);
	assert(batch_normalization_7);

	auto conv2d_8 = network->addConvolutionNd(
		*max_pooling2d_1->getOutput(0), conv_dict["conv2d_8"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_8"].at(0), conv_dict["conv2d_8"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_8_kernel"].data(), conv_dict["conv2d_8"].at(0) * conv_dict["conv2d_8"].at(0) * conv_dict["conv2d_8"].at(1) * conv_dict["conv2d_8"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_8_bias"].data(), conv_dict["conv2d_8"].at(2) }
	);
	assert(conv2d_8);
	conv2d_8->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_8 = network->addScaleNd(
		*conv2d_8->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_8_shift"].data(), bn_dict["batch_normalization_8"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_8_scale"].data(), bn_dict["batch_normalization_8"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_8_power"].data(), bn_dict["batch_normalization_8"] },
		1
	);
	assert(batch_normalization_8);

	auto add_2 = network->addElementWise(
		*batch_normalization_7->getOutput(0),
		*batch_normalization_8->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_2);

	auto activation_5 = network->addActivation(*add_2->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_5);

	auto max_pooling2d_2 = network->addPoolingNd(
		*activation_5->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	assert(max_pooling2d_2);
	/* End of Third CBRA Block*/

	/*
	*	Fourth CBRA Block
	*/
	auto conv2d_9 = network->addConvolutionNd(
		*max_pooling2d_2->getOutput(0), conv_dict["conv2d_9"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_9"].at(0), conv_dict["conv2d_9"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_9_kernel"].data(), conv_dict["conv2d_9"].at(0) * conv_dict["conv2d_9"].at(0) * conv_dict["conv2d_9"].at(1) * conv_dict["conv2d_9"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_9_bias"].data(), conv_dict["conv2d_9"].at(2) }
	);
	assert(conv2d_9);
	conv2d_9->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_9 = network->addScaleNd(
		*conv2d_9->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_9_shift"].data(), bn_dict["batch_normalization_9"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_9_scale"].data(), bn_dict["batch_normalization_9"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_9_power"].data(), bn_dict["batch_normalization_9"] },
		1
	);
	assert(batch_normalization_9);

	auto activation_6 = network->addActivation(
		*batch_normalization_9->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_6);

	auto conv2d_10 = network->addConvolutionNd(
		*activation_6->getOutput(0), conv_dict["conv2d_10"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_10"].at(0), conv_dict["conv2d_10"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_10_kernel"].data(), conv_dict["conv2d_10"].at(0) * conv_dict["conv2d_10"].at(0) * conv_dict["conv2d_10"].at(1) * conv_dict["conv2d_10"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_10_bias"].data(), conv_dict["conv2d_10"].at(2) }
	);
	assert(conv2d_10);
	conv2d_10->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_10 = network->addScaleNd(
		*conv2d_10->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_10_shift"].data(), bn_dict["batch_normalization_10"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_10_scale"].data(), bn_dict["batch_normalization_10"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_10_power"].data(), bn_dict["batch_normalization_10"] },
		1
	);
	assert(batch_normalization_10);

	auto conv2d_11 = network->addConvolutionNd(
		*max_pooling2d_2->getOutput(0), conv_dict["conv2d_11"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_11"].at(0), conv_dict["conv2d_11"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_11_kernel"].data(), conv_dict["conv2d_11"].at(0) * conv_dict["conv2d_11"].at(0) * conv_dict["conv2d_11"].at(1) * conv_dict["conv2d_11"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_11_bias"].data(), conv_dict["conv2d_11"].at(2) }
	);
	assert(conv2d_11);
	conv2d_11->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_11 = network->addScaleNd(
		*conv2d_11->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_11_shift"].data(), bn_dict["batch_normalization_11"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_11_scale"].data(), bn_dict["batch_normalization_11"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_11_power"].data(), bn_dict["batch_normalization_11"] },
		1
	);
	assert(batch_normalization_11);

	auto add_3 = network->addElementWise(
		*batch_normalization_10->getOutput(0),
		*batch_normalization_11->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_3);

	auto activation_7 = network->addActivation(*add_3->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_7);

	auto max_pooling2d_3 = network->addPoolingNd(
		*activation_7->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	assert(max_pooling2d_3);
	/* End of Fourth CBRA Block*/

	/*
	*	1st Bottleneck Block
	*/
	auto conv2d_12 = network->addConvolutionNd(
		*max_pooling2d_3->getOutput(0), conv_dict["conv2d_12"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_12"].at(0), conv_dict["conv2d_12"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_12_kernel"].data(), conv_dict["conv2d_12"].at(0) * conv_dict["conv2d_12"].at(0) * conv_dict["conv2d_12"].at(1) * conv_dict["conv2d_12"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_12_bias"].data(), conv_dict["conv2d_12"].at(2) }
	);
	assert(conv2d_12);
	conv2d_12->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_12 = network->addScaleNd(
		*conv2d_12->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_12_shift"].data(), bn_dict["batch_normalization_12"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_12_scale"].data(), bn_dict["batch_normalization_12"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_12_power"].data(), bn_dict["batch_normalization_12"] },
		1
	);
	assert(batch_normalization_12);

	auto activation_8 = network->addActivation(
		*batch_normalization_12->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_8);

	auto conv2d_13 = network->addConvolutionNd(
		*activation_8->getOutput(0), conv_dict["conv2d_13"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_13"].at(0), conv_dict["conv2d_13"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_13_kernel"].data(), conv_dict["conv2d_13"].at(0) * conv_dict["conv2d_13"].at(0) * conv_dict["conv2d_13"].at(1) * conv_dict["conv2d_13"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_13_bias"].data(), conv_dict["conv2d_13"].at(2) }
	);
	assert(conv2d_13);
	conv2d_13->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_13 = network->addScaleNd(
		*conv2d_13->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_13_shift"].data(), bn_dict["batch_normalization_13"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_13_scale"].data(), bn_dict["batch_normalization_13"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_13_power"].data(), bn_dict["batch_normalization_13"] },
		1
	);
	assert(batch_normalization_13);

	auto conv2d_14 = network->addConvolutionNd(
		*max_pooling2d_3->getOutput(0), conv_dict["conv2d_14"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_14"].at(0), conv_dict["conv2d_14"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_14_kernel"].data(), conv_dict["conv2d_14"].at(0) * conv_dict["conv2d_14"].at(0) * conv_dict["conv2d_14"].at(1) * conv_dict["conv2d_14"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_14_bias"].data(), conv_dict["conv2d_14"].at(2) }
	);
	assert(conv2d_14);
	conv2d_14->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_14 = network->addScaleNd(
		*conv2d_14->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_14_shift"].data(), bn_dict["batch_normalization_14"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_14_scale"].data(), bn_dict["batch_normalization_14"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_14_power"].data(), bn_dict["batch_normalization_14"] },
		1
	);
	assert(batch_normalization_14);

	auto add_4 = network->addElementWise(
		*batch_normalization_13->getOutput(0),
		*batch_normalization_14->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_4);

	auto activation_9 = network->addActivation(*add_4->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_9);
	/* End of 1st Bottleneck Block*/

	/*
	*	2nd Bottleneck Block
	*/
	auto conv2d_15 = network->addConvolutionNd(
		*activation_9->getOutput(0), conv_dict["conv2d_15"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_15"].at(0), conv_dict["conv2d_15"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_15_kernel"].data(), conv_dict["conv2d_15"].at(0) * conv_dict["conv2d_15"].at(0) * conv_dict["conv2d_15"].at(1) * conv_dict["conv2d_15"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_15_bias"].data(), conv_dict["conv2d_15"].at(2) }
	);
	assert(conv2d_15);
	conv2d_15->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_15 = network->addScaleNd(
		*conv2d_15->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_15_shift"].data(), bn_dict["batch_normalization_15"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_15_scale"].data(), bn_dict["batch_normalization_15"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_15_power"].data(), bn_dict["batch_normalization_15"] },
		1
	);
	assert(batch_normalization_15);

	auto activation_10 = network->addActivation(
		*batch_normalization_15->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_10);

	auto conv2d_16 = network->addConvolutionNd(
		*activation_10->getOutput(0), conv_dict["conv2d_16"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_16"].at(0), conv_dict["conv2d_16"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_16_kernel"].data(), conv_dict["conv2d_16"].at(0) * conv_dict["conv2d_16"].at(0) * conv_dict["conv2d_16"].at(1) * conv_dict["conv2d_16"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_16_bias"].data(), conv_dict["conv2d_16"].at(2) }
	);
	assert(conv2d_16);
	conv2d_16->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_16 = network->addScaleNd(
		*conv2d_16->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_16_shift"].data(), bn_dict["batch_normalization_16"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_16_scale"].data(), bn_dict["batch_normalization_16"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_16_power"].data(), bn_dict["batch_normalization_16"] },
		1
	);
	assert(batch_normalization_16);

	auto conv2d_17 = network->addConvolutionNd(
		*activation_9->getOutput(0), conv_dict["conv2d_17"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_17"].at(0), conv_dict["conv2d_17"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_17_kernel"].data(), conv_dict["conv2d_17"].at(0) * conv_dict["conv2d_17"].at(0) * conv_dict["conv2d_17"].at(1) * conv_dict["conv2d_17"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_17_bias"].data(), conv_dict["conv2d_17"].at(2) }
	);
	assert(conv2d_17);
	conv2d_17->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_17 = network->addScaleNd(
		*conv2d_17->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_17_shift"].data(), bn_dict["batch_normalization_17"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_17_scale"].data(), bn_dict["batch_normalization_17"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_17_power"].data(), bn_dict["batch_normalization_17"] },
		1
	);
	assert(batch_normalization_17);

	auto add_5 = network->addElementWise(
		*batch_normalization_16->getOutput(0),
		*batch_normalization_17->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_5);

	auto activation_11 = network->addActivation(*add_5->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_11);
	/* End of 2nd Bottleneck Block*/

	const std::vector<float> up_scales{ 1,1,2,2 };
	/*
	 * 1st Upsampling Block
	 */
	auto up_sampling2d = network->addResize(*activation_11->getOutput(0));
	assert(up_sampling2d);
	up_sampling2d->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_sampling2d->setScales(up_scales.data(), 4);

	std::vector<nvinfer1::ITensor*> concatenate_inputs{ activation_7->getOutput(0), up_sampling2d->getOutput(0) };
	auto concatenate = network->addConcatenation(concatenate_inputs.data(), 2);
	assert(concatenate);

	auto conv2d_18 = network->addConvolutionNd(
		*concatenate->getOutput(0), conv_dict["conv2d_18"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_18"].at(0), conv_dict["conv2d_18"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_18_kernel"].data(), conv_dict["conv2d_18"].at(0) * conv_dict["conv2d_18"].at(0) * conv_dict["conv2d_18"].at(1) * conv_dict["conv2d_18"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_18_bias"].data(), conv_dict["conv2d_18"].at(2) }
	);
	assert(conv2d_18);
	conv2d_18->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_18 = network->addScaleNd(
		*conv2d_18->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_18_shift"].data(), bn_dict["batch_normalization_18"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_18_scale"].data(), bn_dict["batch_normalization_18"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_18_power"].data(), bn_dict["batch_normalization_18"] },
		1
	);
	assert(batch_normalization_18);

	auto activation_12 = network->addActivation(
		*batch_normalization_18->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_12);

	auto conv2d_19 = network->addConvolutionNd(
		*activation_12->getOutput(0), conv_dict["conv2d_19"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_19"].at(0), conv_dict["conv2d_19"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_19_kernel"].data(), conv_dict["conv2d_19"].at(0) * conv_dict["conv2d_19"].at(0) * conv_dict["conv2d_19"].at(1) * conv_dict["conv2d_19"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_19_bias"].data(), conv_dict["conv2d_19"].at(2) }
	);
	assert(conv2d_19);
	conv2d_19->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_19 = network->addScaleNd(
		*conv2d_19->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_19_shift"].data(), bn_dict["batch_normalization_19"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_19_scale"].data(), bn_dict["batch_normalization_19"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_19_power"].data(), bn_dict["batch_normalization_19"] },
		1
	);
	assert(batch_normalization_19);

	auto conv2d_20 = network->addConvolutionNd(
		*concatenate->getOutput(0), conv_dict["conv2d_20"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_20"].at(0), conv_dict["conv2d_20"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_20_kernel"].data(), conv_dict["conv2d_20"].at(0) * conv_dict["conv2d_20"].at(0) * conv_dict["conv2d_20"].at(1) * conv_dict["conv2d_20"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_20_bias"].data(), conv_dict["conv2d_20"].at(2) }
	);
	assert(conv2d_20);
	conv2d_20->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_20 = network->addScaleNd(
		*conv2d_20->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_20_shift"].data(), bn_dict["batch_normalization_20"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_20_scale"].data(), bn_dict["batch_normalization_20"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_20_power"].data(), bn_dict["batch_normalization_20"] },
		1
	);
	assert(batch_normalization_20);

	auto add_6 = network->addElementWise(
		*batch_normalization_19->getOutput(0),
		*batch_normalization_20->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_6);

	auto activation_13 = network->addActivation(*add_6->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_13);
	/* End of 1st Upsampling Block*/

	/*
	 * 2nd Upsampling Block
	 */
	auto up_sampling2d_1 = network->addResize(*activation_13->getOutput(0));
	assert(up_sampling2d_1);
	up_sampling2d_1->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_sampling2d_1->setScales(up_scales.data(), 4);

	std::vector<nvinfer1::ITensor*> concatenate_1_inputs{ activation_5->getOutput(0), up_sampling2d_1->getOutput(0) };
	auto concatenate_1 = network->addConcatenation(concatenate_1_inputs.data(), 2);
	assert(concatenate_1);

	auto conv2d_21 = network->addConvolutionNd(
		*concatenate_1->getOutput(0), conv_dict["conv2d_21"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_21"].at(0), conv_dict["conv2d_21"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_21_kernel"].data(), conv_dict["conv2d_21"].at(0) * conv_dict["conv2d_21"].at(0) * conv_dict["conv2d_21"].at(1) * conv_dict["conv2d_21"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_21_bias"].data(), conv_dict["conv2d_21"].at(2) }
	);
	assert(conv2d_21);
	conv2d_21->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_21 = network->addScaleNd(
		*conv2d_21->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_21_shift"].data(), bn_dict["batch_normalization_21"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_21_scale"].data(), bn_dict["batch_normalization_21"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_21_power"].data(), bn_dict["batch_normalization_21"] },
		1
	);
	assert(batch_normalization_21);

	auto activation_14 = network->addActivation(
		*batch_normalization_21->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_14);

	auto conv2d_22 = network->addConvolutionNd(
		*activation_14->getOutput(0), conv_dict["conv2d_22"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_22"].at(0), conv_dict["conv2d_22"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_22_kernel"].data(), conv_dict["conv2d_22"].at(0) * conv_dict["conv2d_22"].at(0) * conv_dict["conv2d_22"].at(1) * conv_dict["conv2d_22"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_22_bias"].data(), conv_dict["conv2d_22"].at(2) }
	);
	assert(conv2d_22);
	conv2d_22->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_22 = network->addScaleNd(
		*conv2d_22->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_22_shift"].data(), bn_dict["batch_normalization_22"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_22_scale"].data(), bn_dict["batch_normalization_22"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_22_power"].data(), bn_dict["batch_normalization_22"] },
		1
	);
	assert(batch_normalization_22);

	auto conv2d_23 = network->addConvolutionNd(
		*concatenate_1->getOutput(0), conv_dict["conv2d_23"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_23"].at(0), conv_dict["conv2d_23"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_23_kernel"].data(), conv_dict["conv2d_23"].at(0) * conv_dict["conv2d_23"].at(0) * conv_dict["conv2d_23"].at(1) * conv_dict["conv2d_23"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_23_bias"].data(), conv_dict["conv2d_23"].at(2) }
	);
	assert(conv2d_23);
	conv2d_23->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_23 = network->addScaleNd(
		*conv2d_23->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_23_shift"].data(), bn_dict["batch_normalization_23"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_23_scale"].data(), bn_dict["batch_normalization_23"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_23_power"].data(), bn_dict["batch_normalization_23"] },
		1
	);
	assert(batch_normalization_23);

	auto add_7 = network->addElementWise(
		*batch_normalization_22->getOutput(0),
		*batch_normalization_23->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_7);

	auto activation_15 = network->addActivation(*add_7->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_15);
	/* End of 2nd Upsampling Block*/

	/*
	 * 3rd Upsampling Block
	 */
	auto up_sampling2d_2 = network->addResize(*activation_15->getOutput(0));
	assert(up_sampling2d_2);
	up_sampling2d_2->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_sampling2d_2->setScales(up_scales.data(), 4);

	std::vector<nvinfer1::ITensor*> concatenate_2_inputs{ activation_3->getOutput(0), up_sampling2d_2->getOutput(0) };
	auto concatenate_2 = network->addConcatenation(concatenate_2_inputs.data(), 2);
	assert(concatenate_2);

	auto conv2d_24 = network->addConvolutionNd(
		*concatenate_2->getOutput(0), conv_dict["conv2d_24"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_24"].at(0), conv_dict["conv2d_24"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_24_kernel"].data(), conv_dict["conv2d_24"].at(0) * conv_dict["conv2d_24"].at(0) * conv_dict["conv2d_24"].at(1) * conv_dict["conv2d_24"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_24_bias"].data(), conv_dict["conv2d_24"].at(2) }
	);
	assert(conv2d_24);
	conv2d_24->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_24 = network->addScaleNd(
		*conv2d_24->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_24_shift"].data(), bn_dict["batch_normalization_24"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_24_scale"].data(), bn_dict["batch_normalization_24"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_24_power"].data(), bn_dict["batch_normalization_24"] },
		1
	);
	assert(batch_normalization_24);

	auto activation_16 = network->addActivation(
		*batch_normalization_24->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_16);

	auto conv2d_25 = network->addConvolutionNd(
		*activation_16->getOutput(0), conv_dict["conv2d_25"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_25"].at(0), conv_dict["conv2d_25"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_25_kernel"].data(), conv_dict["conv2d_25"].at(0) * conv_dict["conv2d_25"].at(0) * conv_dict["conv2d_25"].at(1) * conv_dict["conv2d_25"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_25_bias"].data(), conv_dict["conv2d_25"].at(2) }
	);
	assert(conv2d_25);
	conv2d_25->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_25 = network->addScaleNd(
		*conv2d_25->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_25_shift"].data(), bn_dict["batch_normalization_25"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_25_scale"].data(), bn_dict["batch_normalization_25"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_25_power"].data(), bn_dict["batch_normalization_25"] },
		1
	);
	assert(batch_normalization_25);

	auto conv2d_26 = network->addConvolutionNd(
		*concatenate_2->getOutput(0), conv_dict["conv2d_26"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_26"].at(0), conv_dict["conv2d_26"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_26_kernel"].data(), conv_dict["conv2d_26"].at(0) * conv_dict["conv2d_26"].at(0) * conv_dict["conv2d_26"].at(1) * conv_dict["conv2d_26"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_26_bias"].data(), conv_dict["conv2d_26"].at(2) }
	);
	assert(conv2d_26);
	conv2d_26->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_26 = network->addScaleNd(
		*conv2d_26->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_26_shift"].data(), bn_dict["batch_normalization_26"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_26_scale"].data(), bn_dict["batch_normalization_26"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_26_power"].data(), bn_dict["batch_normalization_26"] },
		1
	);
	assert(batch_normalization_26);

	auto add_8 = network->addElementWise(
		*batch_normalization_25->getOutput(0),
		*batch_normalization_26->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_8);

	auto activation_17 = network->addActivation(*add_8->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_17);
	/* End of 3rd Upsampling Block*/

	/*
	 * 4th Upsampling Block
	 */
	auto up_sampling2d_3 = network->addResize(*activation_17->getOutput(0));
	assert(up_sampling2d_3);
	up_sampling2d_3->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
	up_sampling2d_3->setScales(up_scales.data(), 4);

	std::vector<nvinfer1::ITensor*> concatenate_3_inputs{ activation_1->getOutput(0), up_sampling2d_3->getOutput(0) };
	auto concatenate_3 = network->addConcatenation(concatenate_3_inputs.data(), 2);
	assert(concatenate_3);

	auto conv2d_27 = network->addConvolutionNd(
		*concatenate_3->getOutput(0), conv_dict["conv2d_27"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_27"].at(0), conv_dict["conv2d_27"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_27_kernel"].data(), conv_dict["conv2d_27"].at(0) * conv_dict["conv2d_27"].at(0) * conv_dict["conv2d_27"].at(1) * conv_dict["conv2d_27"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_27_bias"].data(), conv_dict["conv2d_27"].at(2) }
	);
	assert(conv2d_27);
	conv2d_27->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_27 = network->addScaleNd(
		*conv2d_27->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_27_shift"].data(), bn_dict["batch_normalization_27"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_27_scale"].data(), bn_dict["batch_normalization_27"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_27_power"].data(), bn_dict["batch_normalization_27"] },
		1
	);
	assert(batch_normalization_27);

	auto activation_18 = network->addActivation(
		*batch_normalization_27->getOutput(0), nvinfer1::ActivationType::kRELU
	);
	assert(activation_18);

	auto conv2d_28 = network->addConvolutionNd(
		*activation_18->getOutput(0), conv_dict["conv2d_28"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_28"].at(0), conv_dict["conv2d_28"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_28_kernel"].data(), conv_dict["conv2d_28"].at(0) * conv_dict["conv2d_28"].at(0) * conv_dict["conv2d_28"].at(1) * conv_dict["conv2d_28"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_28_bias"].data(), conv_dict["conv2d_28"].at(2) }
	);
	assert(conv2d_28);
	conv2d_28->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_28 = network->addScaleNd(
		*conv2d_28->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_28_shift"].data(), bn_dict["batch_normalization_28"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_28_scale"].data(), bn_dict["batch_normalization_28"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_28_power"].data(), bn_dict["batch_normalization_28"] },
		1
	);
	assert(batch_normalization_28);

	auto conv2d_29 = network->addConvolutionNd(
		*concatenate_3->getOutput(0), conv_dict["conv2d_29"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_29"].at(0), conv_dict["conv2d_29"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_29_kernel"].data(), conv_dict["conv2d_29"].at(0) * conv_dict["conv2d_29"].at(0) * conv_dict["conv2d_29"].at(1) * conv_dict["conv2d_29"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_29_bias"].data(), conv_dict["conv2d_29"].at(2) }
	);
	assert(conv2d_29);
	conv2d_29->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto batch_normalization_29 = network->addScaleNd(
		*conv2d_29->getOutput(0), nvinfer1::ScaleMode::kCHANNEL,
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_29_shift"].data(), bn_dict["batch_normalization_29"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_29_scale"].data(), bn_dict["batch_normalization_29"] },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["batch_normalization_29_power"].data(), bn_dict["batch_normalization_29"] },
		1
	);
	assert(batch_normalization_29);

	auto add_9 = network->addElementWise(
		*batch_normalization_28->getOutput(0),
		*batch_normalization_29->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_9);

	auto activation_19 = network->addActivation(*add_9->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(activation_19);
	/* End of 4th Upsampling Block*/

	/*
	 * Output Processing
	 */
	auto conv2d_30 = network->addConvolutionNd(
		*activation_19->getOutput(0), conv_dict["conv2d_30"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_30"].at(0), conv_dict["conv2d_30"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_30_kernel"].data(), conv_dict["conv2d_30"].at(0) * conv_dict["conv2d_30"].at(0) * conv_dict["conv2d_30"].at(1) * conv_dict["conv2d_30"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_30_bias"].data(), conv_dict["conv2d_30"].at(2) }
	);
	assert(conv2d_30);
	conv2d_30->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto add_10 = network->addElementWise(
		*conv2d_30->getOutput(0),
		*scaled_input->getOutput(0),
		nvinfer1::ElementWiseOperation::kSUM
	);
	assert(add_10);

	auto conv2d_31 = network->addConvolutionNd(
		*add_10->getOutput(0), conv_dict["conv2d_31"].at(2),
		nvinfer1::DimsHW(conv_dict["conv2d_31"].at(0), conv_dict["conv2d_31"].at(0)),
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_31_kernel"].data(), conv_dict["conv2d_31"].at(0) * conv_dict["conv2d_31"].at(0) * conv_dict["conv2d_31"].at(1) * conv_dict["conv2d_31"].at(2) },
		nvinfer1::Weights{ nvinfer1::DataType::kFLOAT, weight_store["conv2d_31_bias"].data(), conv_dict["conv2d_31"].at(2) }
	);
	assert(conv2d_31);
	conv2d_31->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);

	auto final_output = network->addActivation(
		*conv2d_31->getOutput(0), nvinfer1::ActivationType::kSIGMOID
	);
	assert(final_output);

	auto debug_dims = final_output->getOutput(0)->getDimensions();
	std::cout << debug_dims.d[0] << " x " << debug_dims.d[1] << " x " << debug_dims.d[2] << " x " << debug_dims.d[3] << std::endl;

	//  Set the output
	network->markOutput(*(final_output->getOutput(0)));


	// Build engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	assert(engine != nullptr);

	//  Following sampleMNISTAPI, network is destroyed and host memory for weight is released
	network->destroy();
	std::cout << "UNET V3 Constructed" << std::endl;
	return engine;
}

#endif

