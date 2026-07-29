# Licensing

tinyML is split in two. The core is MIT. The extended model library is
commercial. The split follows the dependency graph, so it is enforceable by
inspection: no MIT header includes a commercial one.

## MIT core

Zero third-party dependencies. Compiler intrinsics (`<immintrin.h>`,
`<arm_neon.h>`) only. Free for any use, including commercial, under
[LICENSE](LICENSE).

| Header | |
| --- | --- |
| `include/common.h` | Shared types and helpers |
| `include/logger.h` | Logging |
| `include/NN.h` | Neural network primitives |
| `include/Network.h` | Network composition |
| `include/Model.h` | Model container |
| `include/Perceptron.h` | Perceptron |
| `include/VectorOperations.h` | Vector maths |
| `include/VectorStatistics.h` | Statistical analysis |
| `include/SIMDOperations.h` | SSE/AVX paths |
| `include/NEONOperations.h` | ARM NEON paths |
| `include/QuantizedOperations.h` | Quantised arithmetic |
| `include/AdvancedOptimizations.h` | Optimiser implementations |
| `include/BayesianNeuralNetwork.h` | Bayesian networks |
| `include/TinyMLAPI.h` | Public API surface |

## Commercial

Requires [xsimd](https://github.com/xtensor-stack/xsimd) (BSD-3-Clause).
Covered by [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). Free for research,
education and personal projects. Commercial use requires a paid licence.

| Header | |
| --- | --- |
| `include/XSIMDOperations.h` | xsimd backend, the root of this tier |
| `include/AdvancedAttention.h` | Attention variants |
| `include/LightweightAttention.h` | Reduced-cost attention |
| `include/RealTimeTransformer.h` | Streaming transformer |
| `include/DynamicNeuralNetwork.h` | Dynamic topology networks |
| `include/GenerativeModels.h` | Generative models |
| `include/GraphNeuralNetwork.h` | Graph networks |
| `include/PhysicsInformedNN.h` | Physics-informed networks |
| `include/ReinforcementLearning.h` | RL agents |
| `include/TimeSeriesForecasting.h` | Forecasting |
| `include/ProductionAPI.h` | Production deployment API |

## Why the split falls here

Commercial headers may include MIT headers. MIT headers include nothing from
the commercial tier. That means the MIT core compiles and ships on its own,
with no third-party dependency and no licence entanglement.

To build core-only, exclude the commercial headers from your include path.
They are not referenced by anything in the core.
