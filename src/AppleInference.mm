// SPDX-License-Identifier: MIT
#include <tinyml/inference.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#import <Foundation/Foundation.h>
#if TINYML_ENABLE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif
#if TINYML_ENABLE_NPU
#import <CoreML/CoreML.h>
#endif

namespace ML::detail
{
    using Runner = std::function<std::vector<double> (const std::vector<double>&)>;

#if TINYML_ENABLE_MPS
    namespace
    {
        struct MPSState
        {
            id<MTLDevice> device;
            id<MTLCommandQueue> queue;
            MPSGraph* graph;
            MPSGraphTensor* input;
            MPSGraphTensor* output;
            unsigned inputWidth = 0;
            unsigned outputWidth = 0;
        };

        float checkedFloat (double value)
        {
            const auto result = static_cast<float> (value);
            if (!std::isfinite (result))
                throw std::invalid_argument ("TinyML MPS requires finite float32 values");
            return result;
        }

        NSData* dataFor (const std::vector<float>& values)
        {
            return [NSData dataWithBytes:values.data() length:values.size() * sizeof(float)];
        }
    }

    bool mpsAvailable() noexcept
    {
        @autoreleasepool
        {
            if (@available(macOS 13.0, *))
                return MTLCreateSystemDefaultDevice() != nil;
            return false;
        }
    }

    Runner makeMPS (const Model& model)
    {
        @autoreleasepool
        {
            if (!mpsAvailable())
                throw std::runtime_error ("TinyML MPS requires a Metal device and macOS 13+");

            auto state = std::make_shared<MPSState>();
            state->device = MTLCreateSystemDefaultDevice();
            state->queue = [state->device newCommandQueue];
            if (!state->queue)
                throw std::runtime_error ("Unable to create Metal command queue");

            state->graph = [[MPSGraph alloc] init];
            state->graph.options = MPSGraphOptionsSynchronizeResults;
            const auto& topology = model.getTopology();
            const auto weights = model.getWeights();
            state->inputWidth = topology.front();
            state->outputWidth = topology.back();
            state->input = [state->graph placeholderWithShape:@[@1, @(state->inputWidth)]
                                                      dataType:MPSDataTypeFloat32 name:@"input"];
            MPSGraphTensor* value = state->input;
            std::size_t offset = 0;
            for (std::size_t layer = 1; layer < topology.size(); ++layer)
            {
                const auto in = topology[layer - 1];
                const auto out = topology[layer];
                std::vector<float> matrix (static_cast<std::size_t> (in) * out);
                std::vector<float> bias (out);
                for (auto& weight : matrix)
                    weight = checkedFloat (weights.at (offset++));
                for (auto& weight : bias)
                    weight = checkedFloat (weights.at (offset++));
                auto* w = [state->graph constantWithData:dataFor (matrix)
                                                 shape:@[@(in), @(out)] dataType:MPSDataTypeFloat32];
                auto* b = [state->graph constantWithData:dataFor (bias)
                                                 shape:@[@1, @(out)] dataType:MPSDataTypeFloat32];
                value = [state->graph matrixMultiplicationWithPrimaryTensor:value secondaryTensor:w name:nil];
                value = [state->graph additionWithPrimaryTensor:value secondaryTensor:b name:nil];
                value = [state->graph tanhWithTensor:value name:nil];
            }
            state->output = value;

            return [state] (const std::vector<double>& input)
            {
                @autoreleasepool
                {
                    if (input.size() != state->inputWidth)
                        throw std::invalid_argument ("TinyML input vector size mismatch");
                    std::vector<float> values;
                    values.reserve (input.size());
                    for (double x : input)
                        values.push_back (checkedFloat (x));
                    id<MTLBuffer> buffer = [state->device newBufferWithBytes:values.data()
                        length:values.size() * sizeof(float) options:MTLResourceStorageModeShared];
                    if (!buffer)
                        throw std::runtime_error ("Unable to allocate MPS input buffer");
                    auto* tensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                        shape:@[@1, @(state->inputWidth)] dataType:MPSDataTypeFloat32];
                    auto* results = [state->graph runWithMTLCommandQueue:state->queue
                        feeds:@{state->input: tensor} targetTensors:@[state->output] targetOperations:nil];
                    auto* result = results[state->output];
                    if (!result)
                        throw std::runtime_error ("MPSGraph produced no output");
                    std::vector<float> output (state->outputWidth);
                    [result.mpsndarray readBytes:output.data() strideBytes:nullptr];
                    return std::vector<double> (output.begin(), output.end());
                }
            };
        }
    }
#endif

#if TINYML_ENABLE_NPU
    namespace
    {
        std::runtime_error coreMLError (const char* message, NSError* error)
        {
            const char* description = error.localizedDescription.UTF8String;
            return std::runtime_error (std::string (message) + (description ? description : ""));
        }

        std::size_t featureWidth (MLFeatureDescription* feature)
        {
            auto* constraint = feature.multiArrayConstraint;
            if (!constraint || constraint.dataType != MLMultiArrayDataTypeFloat32 ||
                constraint.shape.count != 2 || constraint.shape[0].integerValue != 1 ||
                constraint.shape[1].integerValue <= 0)
                throw std::invalid_argument ("Expected a float32 Core ML multiarray of shape [1, width]");
            return static_cast<std::size_t> (constraint.shape[1].integerValue);
        }
    }

    bool coreMLAvailable() noexcept
    {
        if (@available(macOS 13.0, *))
            return true;
        return false;
    }

    Runner makeNPU (const std::string& path)
    {
        @autoreleasepool
        {
            if (@available(macOS 13.0, *))
            {
                auto* filename = [NSString stringWithUTF8String:path.c_str()];
                if (!filename || ![filename.pathExtension isEqualToString:@"mlmodelc"])
                    throw std::invalid_argument ("Expected a compiled .mlmodelc path");
                NSError* error = nil;
                auto* config = [[MLModelConfiguration alloc] init];
                config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                auto* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:filename]
                                               configuration:config error:&error];
                if (!model)
                    throw coreMLError ("Core ML load failed: ", error);
                const auto inputWidth = featureWidth (model.modelDescription.inputDescriptionsByName[@"input"]);
                const auto outputWidth = featureWidth (model.modelDescription.outputDescriptionsByName[@"output"]);

                return [model, inputWidth, outputWidth] (const std::vector<double>& input)
                {
                    @autoreleasepool
                    {
                        if (input.size() != inputWidth)
                            throw std::invalid_argument ("TinyML input vector size mismatch");
                        NSError* error = nil;
                        auto* array = [[MLMultiArray alloc] initWithShape:@[@1, @(inputWidth)]
                            dataType:MLMultiArrayDataTypeFloat32 error:&error];
                        if (!array)
                            throw coreMLError ("Core ML input allocation failed: ", error);
                        for (std::size_t i = 0; i < inputWidth; ++i)
                        {
                            if (!std::isfinite (static_cast<float> (input[i])))
                                throw std::invalid_argument ("TinyML NPU requires finite float32 input");
                            array[@[@0, @(i)]] = @(static_cast<float> (input[i]));
                        }
                        auto* provider = [[MLDictionaryFeatureProvider alloc]
                            initWithDictionary:@{@"input": array} error:&error];
                        if (!provider)
                            throw coreMLError ("Core ML feature creation failed: ", error);
                        id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider error:&error];
                        if (!prediction)
                            throw coreMLError ("Core ML prediction failed: ", error);
                        auto* result = [prediction featureValueForName:@"output"].multiArrayValue;
                        if (!result || result.shape.count != 2 ||
                            result.shape[0].integerValue != 1 ||
                            result.shape[1].unsignedLongLongValue != outputWidth)
                            throw std::runtime_error ("Unexpected Core ML output shape");
                        std::vector<double> output (outputWidth);
                        for (std::size_t i = 0; i < outputWidth; ++i)
                            output[i] = [result[@[@0, @(i)]] doubleValue];
                        return output;
                    }
                };
            }
            throw std::runtime_error ("TinyML NPU requires macOS 13+");
        }
    }
#endif
}
