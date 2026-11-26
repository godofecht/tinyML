# Apple inference SDK (experimental)

Build on macOS 13+:

```sh
cmake -S . -B build -DTINYML_BUILD_EXTENDED=OFF -DTINYML_ENABLE_MPS=ON -DTINYML_ENABLE_NPU=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Both options default to OFF; non-macOS builds remain dependency-free. Link TinyML::Core.
The SDK is an inference-only snapshot of the existing dense tanh network, not a global
switch for training or extended attention/transformer/physics operators.

```cpp
#include <tinyml/core.hpp>
ML::Model model ({2, 3, 1});
// Train or load weights before preparing the session.
ML::InferenceSession cpu (model);
ML::InferenceSession gpu (model, ML::Backend::MPS);
auto result = gpu.predict ({0.25, -0.5});
model.saveWeightsToFile ("weights.txt");
```

MPSGraph uses the entire dense graph with float32 constants and an explicit Metal command
queue. It retains graph and weights across calls, but first-use compilation, input allocation,
synchronization and readback are not hard-real-time safe. No automatic TinyML CPU fallback.
Use `InferenceSession::available(Backend::MPS)` to check for a Metal device.

The public Core ML SDK has no ANE-only compute-unit selection. NPU selects CPU + Neural
Engine, excludes GPU, and may execute any or all operations on CPU. Availability means the
runtime SDK is available, not that an ANE was used. Profile with Xcode Core ML Instruments
or inspect the Core ML compute plan on supported macOS versions.

Export the same topology and text weights offline (no PyTorch dependency):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install 'coremltools>=8,<10' numpy
python scripts/export_coreml.py --topology 2 3 1 --weights weights.txt --output model.mlpackage
mkdir -p compiled
xcrun coremlcompiler compile model.mlpackage compiled
```

```cpp
auto npu = ML::InferenceSession::loadNPU ("compiled/model.mlmodelc");
auto result = npu.predict ({0.25, -0.5});
```

The compiled artifact contains independent weights; export again after training. The exporter
uses float16 internal computation and float32 feature I/O, not bitwise CPU parity. Sessions
are synchronous, fixed batch size one, and not safe to call concurrently. CPU remains the
default and is preferred for very small networks until measured otherwise. GPU/ANE hardware
execution and latency must be verified on a physical Apple Silicon Mac.
