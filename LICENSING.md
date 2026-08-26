# Licensing

tinyML uses a mixed-license repository. Licensing and API stability are separate concepts: [docs/STABILITY.md](docs/STABILITY.md) defines what is supported and installed, while this document defines the license that applies to source files.

## MIT-licensed code

The following files are licensed under [LICENSE](LICENSE), including commercial use under the terms of the MIT License:

`include/common.h`, `include/logger.h`, `include/NN.h`, `include/Network.h`, `include/Model.h`, `include/Perceptron.h`, `include/VectorOperations.h`, `include/VectorStatistics.h`, `include/SIMDOperations.h`, `include/NEONOperations.h`, `include/QuantizedOperations.h`, `include/AdvancedOptimizations.h`, `include/BayesianNeuralNetwork.h`, `include/TinyMLAPI.h` and their corresponding MIT implementation files where present.

The supported `TinyML::Core` distribution deliberately contains only a smaller stable subset: `NN.h`, `Network.h`, `Model.h`, `src/NN.cpp` and `src/Network.cpp`, plus the generated/version umbrella headers. A core-only build does not discover, fetch, link or expose xsimd.

Some historical files contain older copyright banners. Copyright ownership is compatible with an open-source license grant; for files explicitly identified as MIT in this document, [LICENSE](LICENSE) is the repository's license grant for that code.

## Commercially licensed code

The following modules are covered by [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md):

`include/XSIMDOperations.h`, `include/AdvancedAttention.h`, `include/LightweightAttention.h`, `include/RealTimeTransformer.h`, `include/DynamicNeuralNetwork.h`, `include/GenerativeModels.h`, `include/GraphNeuralNetwork.h`, `include/PhysicsInformedNN.h`, `include/ReinforcementLearning.h`, `include/TimeSeriesForecasting.h`, `include/ProductionAPI.h` and their corresponding implementation files.

The `TinyML::Extended` target combines these modules with MIT-licensed support code. Linking MIT code into the extended target does not relicense that MIT code, but use of the commercially licensed modules remains subject to the commercial license.

`TinyMLAPI.h` itself remains MIT licensed, but its current implementation is intentionally part of `TinyML::Extended` because it instantiates extended model types. It is therefore not part of the zero-dependency core artifact.

## Third-party dependency

The extended target uses [xsimd](https://github.com/xtensor-stack/xsimd), which is distributed under the BSD 3-Clause License. Core-only builds do not require xsimd.

## Distribution boundary

The build graph enforces the distribution boundary. `TINYML_BUILD_EXTENDED=OFF` creates and installs only the MIT core target and stable core headers. Enabling the extended build creates a separate `TinyML::Extended` target and extended install export.

The release archives mirror that boundary: the core archive is dependency-free; the extended archive requires an xsimd package when consumed through CMake.
