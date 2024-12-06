#pragma once
#ifndef UNET_ENGINE_H
#define UNET_ENGINE_H

#include <NvInfer.h>
#include <string>
#include <cassert>
#include "h5file.hpp"
#include "custom_layer.h"

/*
	Build the entire network using TensorRT API, following createMNISTEngine() in sampleMNISTAPI.cpp
*/
nvinfer1::ICudaEngine* create_unet_engine_v1(nvinfer1::IBuilder* builder, const char* weight_fname)
{

	//  Create the network to populate the network, then set the outputs and create an engine
	auto network = builder->createNetwork();

	//  Add layers to the network
	auto in_tensor = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::DimsNCHW(1, 1, 896, 896));
	assert(in_tensor != nullptr);

	//  Add Convolutional Layers to the network 

	/*	Block 1
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	*/
	auto conv11_weight = load_weight(weight_fname, "conv2d_24", "kernel:0", 3 * 3 * 64);
	auto conv11_bias = load_weight(weight_fname, "conv2d_24", "bias:0", 64);
	auto conv12_weight = load_weight(weight_fname, "conv2d_25", "kernel:0", 3 * 3 * 64 * 64);
	auto conv12_bias = load_weight(weight_fname, "conv2d_25", "bias:0", 64);
	//  Weights and Bias for first 
	nvinfer1::Weights weights_11{ nvinfer1::DataType::kFLOAT, conv11_weight.data(), 3 * 3 * 64 };
	nvinfer1::Weights bias_11{ nvinfer1::DataType::kFLOAT, conv11_bias.data(), 64 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_12{ nvinfer1::DataType::kFLOAT, conv12_weight.data(), 3 * 3 * 64 * 64 };
	nvinfer1::Weights bias_12{ nvinfer1::DataType::kFLOAT, conv12_bias.data(), 64 };

	auto pad_1_1 = network->addPadding(*in_tensor, nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_1_1 = network->addConvolution(*(pad_1_1->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), weights_11, bias_11);
	auto relu_1_1 = network->addActivation(*(conv_1_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_1_2 = network->addPadding(*(relu_1_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_1_2 = network->addConvolution(*(pad_1_2->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), weights_12, bias_12);
	auto relu_1_2 = network->addActivation(*(conv_1_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_1_3 = network->addPooling(*(relu_1_2->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	auto test_1 = (conv_1_1 && relu_1_1 && conv_1_2 && relu_1_2 && pool_1_3);
	assert(test_1);

	//  pool_1_3 output has dimension (1, 64, 448, 448), seems right


	/*	Block 2
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	*/
	auto conv21_bias = load_weight(weight_fname, "conv2d_26", "bias:0", 128);
	auto conv21_weight = load_weight(weight_fname, "conv2d_26", "kernel:0", 3 * 3 * 64 * 128);
	auto conv22_bias = load_weight(weight_fname, "conv2d_27", "bias:0", 128);
	auto conv22_weight = load_weight(weight_fname, "conv2d_27", "kernel:0", 3 * 3 * 128 * 128);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_21{ nvinfer1::DataType::kFLOAT, conv21_weight.data(), 3 * 3 * 64 * 128 };
	nvinfer1::Weights bias_21{ nvinfer1::DataType::kFLOAT, conv21_bias.data(), 128 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_22{ nvinfer1::DataType::kFLOAT, conv22_weight.data(), 3 * 3 * 128 * 128 };
	nvinfer1::Weights bias_22{ nvinfer1::DataType::kFLOAT, conv22_bias.data(), 128 };

	auto pad_2_1 = network->addPadding(*(pool_1_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_2_1 = network->addConvolution(*(pad_2_1->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), weights_21, bias_21);
	auto relu_2_1 = network->addActivation(*(conv_2_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_2_2 = network->addPadding(*(relu_2_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_2_2 = network->addConvolution(*(pad_2_2->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), weights_22, bias_22);
	auto relu_2_2 = network->addActivation(*(conv_2_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_2_3 = network->addPooling(*(relu_2_2->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	auto test_2 = (conv_2_1 && relu_2_1 && conv_2_2 && relu_2_2 && pool_2_3);
	assert(test_2);

	//  pool_2_3 output has dimension (1, 128, 224, 224), seems right


	/*	Block 3
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	*/
	auto conv31_weight = load_weight(weight_fname, "conv2d_28", "kernel:0", 3 * 3 * 128 * 256);
	auto conv31_bias = load_weight(weight_fname, "conv2d_28", "bias:0", 256);
	auto conv32_weight = load_weight(weight_fname, "conv2d_29", "kernel:0", 3 * 3 * 256 * 256);
	auto conv32_bias = load_weight(weight_fname, "conv2d_29", "bias:0", 256);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_31{ nvinfer1::DataType::kFLOAT, conv31_weight.data(), 3 * 3 * 128 * 256 };
	nvinfer1::Weights bias_31{ nvinfer1::DataType::kFLOAT, conv31_bias.data(), 256 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_32{ nvinfer1::DataType::kFLOAT, conv32_weight.data(), 3 * 3 * 256 * 256 };
	nvinfer1::Weights bias_32{ nvinfer1::DataType::kFLOAT, conv32_bias.data(), 256 };

	auto pad_3_1 = network->addPadding(*(pool_2_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_3_1 = network->addConvolution(*(pad_3_1->getOutput(0)), 256, nvinfer1::DimsHW(3, 3), weights_31, bias_31);
	auto relu_3_1 = network->addActivation(*(conv_3_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_3_2 = network->addPadding(*(relu_3_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_3_2 = network->addConvolution(*(pad_3_2->getOutput(0)), 256, nvinfer1::DimsHW(3, 3), weights_32, bias_32);
	auto relu_3_2 = network->addActivation(*(conv_3_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_3_3 = network->addPooling(*(relu_3_2->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	auto test_3 = (conv_3_1 && relu_3_1 && conv_3_2 && relu_3_2 && pool_3_3);
	assert(test_3);

	//  pool_3_3 has output dimension (1, 256, 112, 122), seems right


	/*	Block 4
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
	*/
	auto conv41_weight = load_weight(weight_fname, "conv2d_30", "kernel:0", 3 * 3 * 256 * 512);
	auto conv41_bias = load_weight(weight_fname, "conv2d_30", "bias:0", 512);
	auto conv42_weight = load_weight(weight_fname, "conv2d_31", "kernel:0", 3 * 3 * 512 * 512);
	auto conv42_bias = load_weight(weight_fname, "conv2d_31", "bias:0", 512);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_41{ nvinfer1::DataType::kFLOAT, conv41_weight.data(), 3 * 3 * 256 * 512 };
	nvinfer1::Weights bias_41{ nvinfer1::DataType::kFLOAT, conv41_bias.data(), 512 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_42{ nvinfer1::DataType::kFLOAT, conv42_weight.data(), 3 * 3 * 512 * 512 };
	nvinfer1::Weights bias_42{ nvinfer1::DataType::kFLOAT, conv42_bias.data(), 512 };
	auto pad_4_1 = network->addPadding(*(pool_3_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_4_1 = network->addConvolution(*(pad_4_1->getOutput(0)), 512, nvinfer1::DimsHW(3, 3), weights_41, bias_41);
	auto relu_4_1 = network->addActivation(*(conv_4_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_4_2 = network->addPadding(*(relu_4_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_4_2 = network->addConvolution(*(pad_4_2->getOutput(0)), 512, nvinfer1::DimsHW(3, 3), weights_42, bias_42);
	auto relu_4_2 = network->addActivation(*(conv_4_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pool_4_3 = network->addPooling(*(relu_4_2->getOutput(0)), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(2, 2));
	auto test_4 = (conv_4_1 && relu_4_1 && conv_4_2 && relu_4_2 && pool_4_3);
	assert(test_4);
	//  we do not need dropout at all?

	//  pool_4_3 has output dimension (1, 512, 56, 56), seems right

	/*	Block 5
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)
	*/
	auto conv51_weight = load_weight(weight_fname, "conv2d_32", "kernel:0", 3 * 3 * 512 * 1024);
	auto conv51_bias = load_weight(weight_fname, "conv2d_32", "bias:0", 1024);
	auto conv52_weight = load_weight(weight_fname, "conv2d_33", "kernel:0", 3 * 3 * 1024 * 1024);
	auto conv52_bias = load_weight(weight_fname, "conv2d_33", "bias:0", 1024);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_51{ nvinfer1::DataType::kFLOAT, conv51_weight.data(), 3 * 3 * 512 * 1024 };
	nvinfer1::Weights bias_51{ nvinfer1::DataType::kFLOAT, conv51_bias.data(), 1024 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_52{ nvinfer1::DataType::kFLOAT, conv52_weight.data(), 3 * 3 * 1024 * 1024 };
	nvinfer1::Weights bias_52{ nvinfer1::DataType::kFLOAT, conv52_bias.data(), 1024 };

	auto pad_5_1 = network->addPadding(*(pool_4_3->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_5_1 = network->addConvolution(*(pad_5_1->getOutput(0)), 1024, nvinfer1::DimsHW(3, 3), weights_51, bias_51);
	auto relu_5_1 = network->addActivation(*(conv_5_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_5_2 = network->addPadding(*(relu_5_1->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_5_2 = network->addConvolution(*(pad_5_2->getOutput(0)), 1024, nvinfer1::DimsHW(3, 3), weights_52, bias_52);
	auto relu_5_2 = network->addActivation(*(conv_5_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto test_5 = (conv_5_1 && relu_5_1 && conv_5_2 && relu_5_2);
	assert(test_5);

	//  relu_5_2 has output dimension (1, 1024, 56, 56), seems right

	/*	Block 6
		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4,up6], axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
	*/
	auto conv61_weight = load_weight(weight_fname, "conv2d_34", "kernel:0", 2 * 2 * 1024 * 512);
	auto conv61_bias = load_weight(weight_fname, "conv2d_34", "bias:0", 512);
	auto conv62_weight = load_weight(weight_fname, "conv2d_35", "kernel:0", 3 * 3 * 1024 * 512);
	auto conv62_bias = load_weight(weight_fname, "conv2d_35", "bias:0", 512);
	auto conv63_weight = load_weight(weight_fname, "conv2d_36", "kernel:0", 3 * 3 * 512 * 512);
	auto conv63_bias = load_weight(weight_fname, "conv2d_36", "bias:0", 512);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_61{ nvinfer1::DataType::kFLOAT, conv61_weight.data(), 2 * 2 * 1024 * 512 };
	nvinfer1::Weights bias_61{ nvinfer1::DataType::kFLOAT, conv61_bias.data(), 512 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_62{ nvinfer1::DataType::kFLOAT, conv62_weight.data(), 3 * 3 * 1024 * 512 };
	nvinfer1::Weights bias_62{ nvinfer1::DataType::kFLOAT, conv62_bias.data(), 512 };
	//  Weights and Bias for third conv
	nvinfer1::Weights weights_63{ nvinfer1::DataType::kFLOAT, conv63_weight.data(), 3 * 3 * 512 * 512 };
	nvinfer1::Weights bias_63{ nvinfer1::DataType::kFLOAT, conv63_bias.data(), 512 };

	UpsampleBy2* upsample_plugin_6 = new UpsampleBy2(56, 56, 1024);
	auto up6_input = relu_5_2->getOutput(0);
	auto up_6 = network->addPluginV2(&up6_input, 1, *upsample_plugin_6);
	auto pad_6_1 = network->addPadding(*(up_6->getOutput(0)), nvinfer1::DimsHW(0, 0), nvinfer1::DimsHW(1, 1));
	auto conv_6_1 = network->addConvolution(*(pad_6_1->getOutput(0)), 512, nvinfer1::DimsHW(2, 2), weights_61, bias_61);
	auto relu_6_1 = network->addActivation(*(conv_6_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	nvinfer1::ITensor** concat6_inputs = new nvinfer1::ITensor * [2];
	concat6_inputs[0] = relu_4_2->getOutput(0);	// drop 4
	concat6_inputs[1] = relu_6_1->getOutput(0); // up6
	auto merge_6 = network->addConcatenation(concat6_inputs, 2);
	auto pad_6_2 = network->addPadding(*(merge_6->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_6_2 = network->addConvolution(*(pad_6_2->getOutput(0)), 512, nvinfer1::DimsHW(3, 3), weights_62, bias_62);
	auto relu_6_2 = network->addActivation(*(conv_6_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_6_3 = network->addPadding(*(relu_6_2->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_6_3 = network->addConvolution(*(pad_6_3->getOutput(0)), 512, nvinfer1::DimsHW(3, 3), weights_63, bias_63);
	auto relu_6_3 = network->addActivation(*(conv_6_3->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto test_6 = (up_6 && conv_6_1 && relu_6_1 && merge_6 && conv_6_2 && relu_6_2 && conv_6_3 && relu_6_3);
	assert(test_6);

	//  relu_6_3 has output dimension (1, 512, 112, 112), seems right

	/*	Block 7
		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
	*/
	auto conv71_weight = load_weight(weight_fname, "conv2d_37", "kernel:0", 2 * 2 * 512 * 256);
	auto conv71_bias = load_weight(weight_fname, "conv2d_37", "bias:0", 256);
	auto conv72_weight = load_weight(weight_fname, "conv2d_38", "kernel:0", 3 * 3 * 512 * 256);
	auto conv72_bias = load_weight(weight_fname, "conv2d_38", "bias:0", 256);
	auto conv73_weight = load_weight(weight_fname, "conv2d_39", "kernel:0", 3 * 3 * 256 * 256);
	auto conv73_bias = load_weight(weight_fname, "conv2d_39", "bias:0", 256);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_71{ nvinfer1::DataType::kFLOAT, conv71_weight.data(), 2 * 2 * 512 * 256 };
	nvinfer1::Weights bias_71{ nvinfer1::DataType::kFLOAT, conv71_bias.data(), 256 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_72{ nvinfer1::DataType::kFLOAT, conv72_weight.data(), 3 * 3 * 512 * 256 };
	nvinfer1::Weights bias_72{ nvinfer1::DataType::kFLOAT, conv72_bias.data(), 256 };
	//  Weights and Bias for third conv
	nvinfer1::Weights weights_73{ nvinfer1::DataType::kFLOAT, conv73_weight.data(), 3 * 3 * 256 * 256 };
	nvinfer1::Weights bias_73{ nvinfer1::DataType::kFLOAT, conv73_bias.data(), 256 };

	UpsampleBy2* upsample_plugin_7 = new UpsampleBy2(112, 112, 512);
	auto up7_input = relu_6_3->getOutput(0);
	auto up_7 = network->addPluginV2(&up7_input, 1, *upsample_plugin_7);
	auto pad_7_1 = network->addPadding(*(up_7->getOutput(0)), nvinfer1::DimsHW(0, 0), nvinfer1::DimsHW(1, 1));
	auto conv_7_1 = network->addConvolution(*(pad_7_1->getOutput(0)), 256, nvinfer1::DimsHW(2, 2), weights_71, bias_71);
	auto relu_7_1 = network->addActivation(*(conv_7_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	nvinfer1::ITensor** concat7_inputs = new nvinfer1::ITensor * [2];
	concat7_inputs[0] = relu_3_2->getOutput(0);	// conv3
	concat7_inputs[1] = relu_7_1->getOutput(0); // up7
	auto merge_7 = network->addConcatenation(concat7_inputs, 2);
	auto pad_7_2 = network->addPadding(*(merge_7->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_7_2 = network->addConvolution(*(pad_7_2->getOutput(0)), 256, nvinfer1::DimsHW(3, 3), weights_72, bias_72);
	auto relu_7_2 = network->addActivation(*(conv_7_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_7_3 = network->addPadding(*(relu_7_2->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_7_3 = network->addConvolution(*(pad_7_3->getOutput(0)), 256, nvinfer1::DimsHW(3, 3), weights_73, bias_73);
	auto relu_7_3 = network->addActivation(*(conv_7_3->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto test_7 = (up_7 && conv_7_1 && relu_7_1 && merge_7 && conv_7_2 && relu_7_2 && conv_7_3 && relu_7_3);
	assert(test_7);

	// relu_7_3 has output dimension (1,256,224,224), seems right

	/*	Block 8
		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
	*/
	auto conv81_weight = load_weight(weight_fname, "conv2d_40", "kernel:0", 2 * 2 * 256 * 128);
	auto conv81_bias = load_weight(weight_fname, "conv2d_40", "bias:0", 128);
	auto conv82_weight = load_weight(weight_fname, "conv2d_41", "kernel:0", 3 * 3 * 256 * 128);
	auto conv82_bias = load_weight(weight_fname, "conv2d_41", "bias:0", 128);
	auto conv83_weight = load_weight(weight_fname, "conv2d_42", "kernel:0", 3 * 3 * 128 * 128);
	auto conv83_bias = load_weight(weight_fname, "conv2d_42", "bias:0", 128);
	//  Weights and Bias for first conv
	nvinfer1::Weights weights_81{ nvinfer1::DataType::kFLOAT, conv81_weight.data(), 2 * 2 * 256 * 128 };
	nvinfer1::Weights bias_81{ nvinfer1::DataType::kFLOAT, conv81_bias.data(), 128 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_82{ nvinfer1::DataType::kFLOAT, conv82_weight.data(), 3 * 3 * 256 * 128 };
	nvinfer1::Weights bias_82{ nvinfer1::DataType::kFLOAT, conv82_bias.data(), 128 };
	//  Weights and Bias for third conv
	nvinfer1::Weights weights_83{ nvinfer1::DataType::kFLOAT, conv83_weight.data(), 3 * 3 * 128 * 128 };
	nvinfer1::Weights bias_83{ nvinfer1::DataType::kFLOAT, conv83_bias.data(), 128 };

	UpsampleBy2* upsample_plugin_8 = new UpsampleBy2(224, 224, 256);
	auto up8_input = relu_7_3->getOutput(0);
	auto up_8 = network->addPluginV2(&up8_input, 1, *upsample_plugin_8);
	auto pad_8_1 = network->addPadding(*(up_8->getOutput(0)), nvinfer1::DimsHW(0, 0), nvinfer1::DimsHW(1, 1));
	auto conv_8_1 = network->addConvolution(*(pad_8_1->getOutput(0)), 128, nvinfer1::DimsHW(2, 2), weights_81, bias_81);
	auto relu_8_1 = network->addActivation(*(conv_8_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	nvinfer1::ITensor** concat8_inputs = new nvinfer1::ITensor * [2];
	concat8_inputs[0] = relu_2_2->getOutput(0); // conv2
	concat8_inputs[1] = relu_8_1->getOutput(0); // up8
	auto merge_8 = network->addConcatenation(concat8_inputs, 2);
	auto pad_8_2 = network->addPadding(*(merge_8->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_8_2 = network->addConvolution(*(pad_8_2->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), weights_82, bias_82);
	auto relu_8_2 = network->addActivation(*(conv_8_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_8_3 = network->addPadding(*(relu_8_2->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_8_3 = network->addConvolution(*(pad_8_3->getOutput(0)), 128, nvinfer1::DimsHW(3, 3), weights_83, bias_83);
	auto relu_8_3 = network->addActivation(*(conv_8_3->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto test_8 = (up_8 && conv_8_1 && relu_8_1 && merge_8 && conv_8_2 && relu_8_2 && conv_8_3 && relu_8_3);
	assert(test_8);

	//  relu_8_3 has output dimension (1, 128, 448, 448), seems right

	/*	Block 9
		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	*/
	auto conv91_weight = load_weight(weight_fname, "conv2d_43", "kernel:0", 2 * 2 * 128 * 64);
	auto conv91_bias = load_weight(weight_fname, "conv2d_43", "bias:0", 64);
	auto conv92_weight = load_weight(weight_fname, "conv2d_44", "kernel:0", 3 * 3 * 128 * 64);
	auto conv92_bias = load_weight(weight_fname, "conv2d_44", "bias:0", 64);
	auto conv93_weight = load_weight(weight_fname, "conv2d_45", "kernel:0", 3 * 3 * 64 * 64);
	auto conv93_bias = load_weight(weight_fname, "conv2d_45", "bias:0", 64);

	//  Weights and Bias for first conv
	nvinfer1::Weights weights_91{ nvinfer1::DataType::kFLOAT, conv91_weight.data(), 2 * 2 * 128 * 64 };
	nvinfer1::Weights bias_91{ nvinfer1::DataType::kFLOAT, conv91_bias.data(), 64 };
	//  Weights and Bias for second conv
	nvinfer1::Weights weights_92{ nvinfer1::DataType::kFLOAT, conv92_weight.data(), 3 * 3 * 128 * 64 };
	nvinfer1::Weights bias_92{ nvinfer1::DataType::kFLOAT, conv92_bias.data(), 64 };
	//  Weights and Bias for third conv
	nvinfer1::Weights weights_93{ nvinfer1::DataType::kFLOAT, conv93_weight.data(), 3 * 3 * 64 * 64 };
	nvinfer1::Weights bias_93{ nvinfer1::DataType::kFLOAT, conv93_bias.data(), 64 };

	UpsampleBy2* upsample_plugin_9 = new UpsampleBy2(448, 448, 128);
	auto up9_input = relu_8_3->getOutput(0);
	auto up_9 = network->addPluginV2(&up9_input, 1, *upsample_plugin_9);
	auto pad_9_1 = network->addPadding(*(up_9->getOutput(0)), nvinfer1::DimsHW(0, 0), nvinfer1::DimsHW(1, 1));
	auto conv_9_1 = network->addConvolution(*(pad_9_1->getOutput(0)), 64, nvinfer1::DimsHW(2, 2), weights_91, bias_91);
	auto relu_9_1 = network->addActivation(*(conv_9_1->getOutput(0)), nvinfer1::ActivationType::kRELU);
	nvinfer1::ITensor** concat9_inputs = new nvinfer1::ITensor * [2];
	concat9_inputs[0] = relu_1_2->getOutput(0); // conv1
	concat9_inputs[1] = relu_9_1->getOutput(0); // up9
	auto merge_9 = network->addConcatenation(concat9_inputs, 2);
	auto pad_9_2 = network->addPadding(*(merge_9->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_9_2 = network->addConvolution(*(pad_9_2->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), weights_92, bias_92);
	auto relu_9_2 = network->addActivation(*(conv_9_2->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto pad_9_3 = network->addPadding(*(relu_9_2->getOutput(0)), nvinfer1::DimsHW(1, 1), nvinfer1::DimsHW(1, 1));
	auto conv_9_3 = network->addConvolution(*(pad_9_3->getOutput(0)), 64, nvinfer1::DimsHW(3, 3), weights_93, bias_93);
	auto relu_9_3 = network->addActivation(*(conv_9_3->getOutput(0)), nvinfer1::ActivationType::kRELU);
	auto test_9 = (up_9 && conv_9_1 && relu_9_1 && merge_9 && conv_9_2 && relu_9_2 && conv_9_3 && relu_9_3);
	assert(test_9);

	//  relu_9_3 has output dimension (1, 64, 896, 896), seems right

	/*	Block 10
		conv10 = Conv2D(4, 1, activation = 'sigmoid')(conv9)
	*/
	auto conv101_weight = load_weight(weight_fname, "conv2d_46", "kernel:0", 1 * 1 * 64 * 4);
	auto conv101_bias = load_weight(weight_fname, "conv2d_46", "bias:0", 4);
	nvinfer1::Weights weights_101{ nvinfer1::DataType::kFLOAT, conv101_weight.data(), 1 * 1 * 64 * 4 };
	nvinfer1::Weights bias_101{ nvinfer1::DataType::kFLOAT, conv101_bias.data(), 4 };

	auto conv_10_1 = network->addConvolution(*(relu_9_3->getOutput(0)), 4, nvinfer1::DimsHW(1, 1), weights_101, bias_101);
	auto relu_10_1 = network->addActivation(*(conv_10_1->getOutput(0)), nvinfer1::ActivationType::kSIGMOID);

	auto out_dimension = relu_10_1->getOutput(0)->getDimensions();

	//  relu_10_1 has output dimension (1, 4, 896, 896), seems right

	//  Set the output
	network->markOutput(*(relu_10_1->getOutput(0)));
	//network->markOutput(*(pool_1_3->getOutput(0)));

	auto layers_in_network = network->getNbLayers();

	// Build engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);
	//builder->setFp16Mode(gArgs.runInFp16);

	auto engine = builder->buildCudaEngine(*network);
	assert(engine != nullptr);

	//  Following sampleMNISTAPI, network is destroyed and host memory for weight is released
	network->destroy();


	return engine;
}

#endif