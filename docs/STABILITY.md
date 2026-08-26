# API stability

The repository intentionally separates licensing, buildability and API stability. A file can be MIT licensed without being part of the supported core API, and a source file can remain in the repository without being installed into a release SDK.

## Stable

The stable surface is the `TinyML::Core` CMake target and the headers installed by a core-only package:

| Header | Contract |
| --- | --- |
| `<tinyml/core.hpp>` | Stable umbrella header |
| `<tinyml/version.hpp>` | Stable version macros |
| `<NN.h>` | Neuron and layer primitives |
| `<Network.h>` | Feed-forward network composition and training |
| `<Model.h>` | Model wrapper, persistence and inference |

Stable APIs follow semantic versioning. Source-compatible additions may land in minor releases. Breaking source or behavior changes require a new major version unless they fix undefined behavior, memory safety, data corruption or another correctness defect that cannot reasonably be preserved.

## Preview

The `TinyML::Extended` target is installable and packaged, but its module-level APIs are currently preview surfaces:

`AdvancedAttention.h`, `DynamicNeuralNetwork.h`, `GenerativeModels.h`, `GraphNeuralNetwork.h`, `LightweightAttention.h`, `PhysicsInformedNN.h`, `ProductionAPI.h`, `QuantizedOperations.h`, `RealTimeTransformer.h`, `ReinforcementLearning.h`, `TimeSeriesForecasting.h`, `TinyMLAPI.h` and `XSIMDOperations.h`.

Preview modules are compiled in the normal extended build and their supported tests run in CI. Their public APIs may still change in minor releases while they are hardened. Graduation to stable requires deterministic correctness tests, no disabled known-failure test, documented serialization behavior where applicable, and inclusion in the installed-consumer test matrix.

## Source-only / experimental

The following legacy or incomplete modules remain available to repository developers but are not installed as part of the supported SDK:

`AdvancedOptimizations.h`, `BayesianNeuralNetwork.h`, `NEONOperations.h`, `Perceptron.h`, `SIMDOperations.h`, `VectorOperations.h`, `VectorStatistics.h`, `common.h` and `logger.h`.

These files are not covered by API compatibility guarantees. Keeping them source-only prevents a partially implemented symbol or historical helper API from accidentally becoming permanent public surface area.

## Test status is part of stability

A module with a disabled test is not considered stable. Long-running experiments may remain disabled in CI for cost reasons, but a module cannot graduate until it also has a deterministic fast correctness suite appropriate for release gating.

Benchmarks are evidence, not tests. Wall-clock thresholds are never used as correctness gates on shared CI hardware.
