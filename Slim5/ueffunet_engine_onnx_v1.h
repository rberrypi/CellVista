#pragma once
#ifndef UEFFNET_ENGINE_ONNX_V1_H
#define UEFFNET_ENGINE_ONNX_V1_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cassert>
#include "ml_shared.h"
#include "qli_runtime_error.h"
#include <iostream>
#include <string>

class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) override
	{
#if _DEBUG
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
#endif
	}
};


// https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_c
nvinfer1::ICudaEngine* buildONNXModel(nvinfer1::IBuilder* builder, const char* model_fname, Logger gLogger)
{

	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	assert(network);
	auto parser = nvonnxparser::createParser(*network, gLogger);
	assert(parser);

	auto parsed = parser->parseFromFile(model_fname, 1); // 1 error 2 warning

	std::cout << "------- ONNX Parsing Errors ? ------" << std::endl;
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}
	assert(parsed);


	// Build engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 30);
	// 20 doesn't seem to work so I increased to 30 (1GB)
	// Related: https://forums.developer.nvidia.com/t/could-not-find-any-implementation-for-node-2-layer-mlp-try-increasing-the-workspace-size-with-ibuilder-setmaxworkspacesize/66503/3
	// Related: https://github.com/NVIDIA/TensorRT/issues/209
	// Related: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#error-messaging

	std::cout << "Start building Cuda Engine from ONNX model" << std::endl;
	auto engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "Done building Cuda Engine from ONNX model" << std::endl;

	//  Following sampleMNISTAPI, network is destroyed and host memory for weight is released
	parser->destroy();
	network->destroy();
	std::cout << "Enginer created, network and parser destroyed" << std::endl;
	return engine;
}

#endif