# tinyML

A lightweight C++ machine-learning and statistics library for embedded, edge and real-time workloads.

## What this is

tinyML is a library (a collection of code other programs can use) for building neural networks in C++. A neural network is a set of connected artificial neurons that learns to turn inputs into outputs. "Embedded" and "edge" mean small computers built into devices, such as a microcontroller in a sensor, where there is no room for a large framework like PyTorch.

The library has two parts:

- **Core** (`TinyML::Core`): a small, stable neural network with no other dependencies. It is MIT licensed, so anyone can use it.
- **Extended** (`TinyML::Extended`): research modules built on top, including attention and transformers, graph neural networks, physics-informed networks, Bayesian networks, reinforcement learning, time-series forecasting, generative models, quantisation and SIMD maths. Some of these files are under a separate commercial licence (see [Licence](#licence)).

The library has no window of its own. You write a small program that uses it, and that program prints results. There is also a playground, a small web page for trying the extended models (see [Technical details](#playground)).

## What you can learn from it

- Computer Science: what a neuron, a weight and a layer are, in readable C++.
- Computer Science: how backpropagation trains a network, shown on the XOR problem below.
- Computer Science: how a C++ library is built, tested, installed and used by another project with CMake.
- Computer Science: SIMD (single instruction, multiple data), where the processor does the same sum on several numbers at once, and quantisation, where numbers are stored with fewer bits to save memory.
- Software engineering: semantic versioning, automated tests and the difference between a stable and a preview API.

## Run it

### On your own computer

**You will need:**

- A C++17 compiler: Xcode Command Line Tools on macOS, `build-essential` on Ubuntu, or [Visual Studio](https://visualstudio.microsoft.com/) with the C++ workload on Windows.
- [CMake](https://cmake.org/download/) 3.20 or newer.
- [Git](https://git-scm.com/downloads).

The project's automatic checks build and test it on Ubuntu with GCC and Clang. The core build, its test and the XOR example below were run on macOS for this guide.

1. Download the code:

   ```bash
   git clone https://github.com/godofecht/tinyML.git
   cd tinyML
   ```

2. Build the core and run its test:

   ```bash
   cmake -S . -B build -DTINYML_BUILD_EXTENDED=OFF
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure
   ```

   You should see `TinyMLCoreSmoke` pass and a line that starts `100% tests passed`.

3. Write your first program. Save this as `xor.cpp` in the `tinyML` folder. It teaches a network XOR: output 1 when exactly one of the two inputs is 1, and 0 otherwise.

   ```cpp
   #include <tinyml/core.hpp>
   #include <iostream>

   int main() {
       ML::Model model({2, 4, 1});   // 2 inputs, 4 hidden neurons, 1 output
       const double in[4][2] = {{0,0},{0,1},{1,0},{1,1}};
       const double out[4] = {0,1,1,0};

       for (int pass = 0; pass < 5000; ++pass)
           for (int i = 0; i < 4; ++i) {
               model.feedForward({in[i][0], in[i][1]});
               model.backPropagate({out[i]});
           }

       for (int i = 0; i < 4; ++i) {
           model.feedForward({in[i][0], in[i][1]});
           std::cout << in[i][0] << " XOR " << in[i][1]
                     << " -> " << model.getResult()[0] << "\n";
       }
   }
   ```

4. Compile and run it:

   ```bash
   c++ -std=c++17 -Iinclude -Ibuild/generated src/NN.cpp src/Network.cpp xor.cpp -o xor
   ./xor
   ```

   You should see four lines. The answers for 0 XOR 1 and 1 XOR 0 should be close to 1 (around 0.99), and the other two close to 0. The exact numbers change from run to run because the starting weights are random.

To build everything, including the extended modules and their tests, leave out `-DTINYML_BUILD_EXTENDED=OFF` in step 2. CMake then downloads [xsimd](https://github.com/xtensor-stack/xsimd) and [GoogleTest](https://github.com/google/googletest), so it needs an internet connection and takes longer.

## How it works

A neuron adds up the outputs of the neurons in the layer before it, each multiplied by a weight (a number that sets how much that input matters). It passes the total through tanh, which squashes any number into the range -1 to 1. Each layer also has a bias neuron whose output is always 1, which lets a neuron shift its total up or down.

The network learns by backpropagation. After `feedForward` works out the outputs, `backPropagate` compares them with the targets, works out how much each weight added to the error, and moves each weight a small step to reduce it. The step size is the learning rate (eta, 0.15). Momentum (alpha, 0.5) adds half of the previous step, which keeps the weights moving in a steady direction.

`ML::Model` wraps a network and adds saving and loading of weights to a text file.

| File or folder | What is in it |
| --- | --- |
| `include/NN.h`, `src/NN.cpp` | the neuron: weights, tanh, gradients, weight updates |
| `include/Network.h`, `src/Network.cpp` | layers of neurons, feed-forward and backpropagation |
| `include/Model.h` | the `ML::Model` wrapper, with `saveWeightsToFile` and `loadWeightsFromFile` |
| `include/tinyml/core.hpp` | the one header to include for the core |
| `include/`, `src/` (other files) | the extended modules |
| `tests/` | the core smoke test, GoogleTest suites and the install test |
| `benchmarks/` | speed measurements, built only on request |
| `examples/` | example programs, built only on request |
| `playground/` | a small web server and page for trying models in a browser |
| `blog/` | write-ups of each development phase |
| `docs/` | the API stability policy, production API notes and a wiki page |
| `CMakeLists.txt` | the build |

## Ideas to try

1. In `xor.cpp`, change the hidden layer from 4 neurons to 2, then to 1. How many does XOR need?
2. In `include/NN.h`, change `eta` and `alpha` and rebuild `xor.cpp`. How many passes does XOR need now?
3. In `src/NN.cpp`, `transferFunction` uses tanh. Try the sigmoid function, 1 / (1 + e^-x), instead. Remember to change `transferFunctionDerivative` to match.
4. Use `saveWeightsToFile` and `loadWeightsFromFile` from `include/Model.h` to save a trained XOR network and load it into a new model without training.
5. Add a check to `tests/core_smoke.cpp` that a model learns XOR, so the test suite fails if training ever breaks.
6. Read `blog/phase-2-attention-mechanism-core.md`, then find the matching code in `src/LightweightAttention.cpp`.

## Technical details

The repository contains two intentionally different surfaces. `TinyML::Core` is the supported, dependency-free library surface with an install/export contract and semantic-versioning guarantees. `TinyML::Extended` contains the xsimd-backed model stack and research modules. The extended target is built by default for source compatibility, but its individual modules have their own stability status documented in [docs/STABILITY.md](docs/STABILITY.md).

### Requirements

The core requires a C++17 compiler and CMake 3.20 or newer. It has no third-party runtime or build dependency.

The extended target additionally requires xsimd 12.1 or newer. Top-level source builds can fetch xsimd automatically; package consumers are expected to provide it through their normal dependency manager.

### Build

Core only:

```bash
cmake -S . -B build -DTINYML_BUILD_EXTENDED=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Full source build:

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Benchmarks, examples and the playground are excluded from ordinary library builds on purpose. Enable them explicitly with `TINYML_BUILD_BENCHMARKS`, `TINYML_BUILD_EXAMPLES` or `TINYML_BUILD_PLAYGROUND`. All three need the extended build.

### Install and consume

```bash
cmake -S . -B build -DTINYML_BUILD_EXTENDED=OFF -DTINYML_BUILD_TESTS=OFF
cmake --build build --parallel
cmake --install build --prefix ./install
```

A downstream CMake project can then use the installed core without xsimd:

```cmake
find_package(TinyML 1.0 CONFIG REQUIRED COMPONENTS Core)
target_link_libraries(my_target PRIVATE TinyML::Core)
```

```cpp
#include <tinyml/core.hpp>

ML::Model model ({ 2, 4, 1 });
model.feedForward ({ 0.25, -0.5 });
auto output = model.getResult();
```

For the extended package:

```cmake
find_package(TinyML 1.0 CONFIG REQUIRED COMPONENTS Extended)
target_link_libraries(my_target PRIVATE TinyML::Extended)
```

The installed package includes a generated `<tinyml/version.hpp>` with `TINYML_VERSION_MAJOR`, `TINYML_VERSION_MINOR`, `TINYML_VERSION_PATCH` and `TINYML_VERSION_STRING`.

### Build options

| Option | Default | Purpose |
| --- | --- | --- |
| `TINYML_BUILD_EXTENDED` | `ON` | Build the xsimd-backed extended library |
| `TINYML_BUILD_TESTS` | top-level only | Build the test suite |
| `TINYML_BUILD_BENCHMARKS` | `OFF` | Build local benchmark executables |
| `TINYML_BUILD_EXAMPLES` | `OFF` | Build examples |
| `TINYML_BUILD_PLAYGROUND` | `OFF` | Build the playground server |
| `TINYML_FETCH_DEPENDENCIES` | top-level only | Fetch missing xsimd/GoogleTest dependencies |
| `TINYML_ENABLE_WARNINGS` | top-level only | Enable compiler warning flags on TinyML targets |
| `TINYML_ENABLE_LTO` | `OFF` | Enable IPO/LTO when supported |

### Playground

The playground is a small web server (`playground/server.cpp`, using cpp-httplib and nlohmann/json) with a browser page that runs and trains models through `/run/<model>` and `/train/<model>`. It listens on port 8080. These steps follow the build file and have not been tested:

```bash
cmake -S . -B build -DTINYML_BUILD_PLAYGROUND=ON
cmake --build build --target PlaygroundServer
./build/PlaygroundServer
```

Then open `http://localhost:8080/playground/index.html`. The server shares the whole repository folder on every network interface, so only run it on a network you trust.

### Testing and release guarantees

CI builds the zero-dependency core separately from the full library with GCC and Clang. Every configuration is installed into a staging prefix and then consumed by a fresh downstream CMake project through `find_package`, so packaging regressions fail before release.

Timing assertions are opt-in because shared CI hardware is unsuitable for performance gates. Benchmark executables are local-only. See [TESTING.md](TESTING.md).

Versioned releases are assembled from `cmake --install` instead of by manually copying build artifacts. Core and extended archives are published separately, with SHA-256 checksums.

### API stability

The compatibility contract is written down in [docs/STABILITY.md](docs/STABILITY.md), which lists the stable, preview and source-only surfaces. The presence of a header in the repository does not by itself make it part of the contract.

Changes to the stable core follow semantic versioning. Preview and source-only modules may change before they graduate into the stable surface.

### Other documents

Security reports should follow [SECURITY.md](SECURITY.md). Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md). Release history is tracked in [CHANGELOG.md](CHANGELOG.md). Planned work is in [ROADMAP.md](ROADMAP.md) and [docs/DEMO_ROADMAP.md](docs/DEMO_ROADMAP.md).

## Project status

[![CI](https://github.com/godofecht/tinyML/actions/workflows/ci.yml/badge.svg)](https://github.com/godofecht/tinyML/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/godofecht/tinyML/actions/workflows/pages.yml/badge.svg)](https://github.com/godofecht/tinyML/actions/workflows/pages.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17%2B-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

Version 1.0.0. The core is stable. The extended modules are previews and may change.

Known problems:

- Three tests are compiled but turned off: `Phase8PhysicsInformedTest` (long-running), `Phase8PhysicsComprehensiveTest` (results vary between runs) and `Phase12ReinforcementTest` (known failure on Linux).
- The GitHub Pages workflow, which publishes `site/`, has failed on its recent runs, so there is no live project page.
- `scripts/run_playground.sh` does not work as written. It does not turn on `TINYML_BUILD_PLAYGROUND`, and it looks for the server in `build/bin/`, where the build does not put it. Use the playground steps above instead.
- The v1.0.0 release on GitHub has no downloadable archives. The packaging described above is listed under "Unreleased" in `CHANGELOG.md`.

## Licence

tinyML uses two licences. Which one applies depends on the file. [LICENSING.md](LICENSING.md) lists every file.

- **The core and some support files are MIT licensed.** You can use, copy, change, share and sell them, as long as you keep the copyright notice. This covers `NN.h`, `Network.h`, `Model.h` and their source files, and others listed in LICENSING.md. See [LICENSE](LICENSE) (MIT).
- **The commercial tier is proprietary.** This covers the attention, transformer, dynamic network, generative, graph, physics-informed, reinforcement learning, time-series, production API and xsimd modules. You may use them for free, for non-commercial research, education or personal projects. You may also evaluate them internally, for non-commercial purposes, for up to 30 days. Any use must credit "Powered by tinyML (c) Abhishek Shivakumar". Commercial use needs a paid licence (USD 199 per developer per year). You may not share these files, or changed versions of them, without written permission. See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

For a school project this means you can use any part of tinyML, as long as you add the credit line. If you publish your work, publish only your own code and the MIT files, and link to this repo for the rest.

## Credits

- Written by Abhishek Shivakumar.
- [xsimd](https://github.com/xtensor-stack/xsimd) (BSD 3-Clause) for SIMD in the extended build.
- [GoogleTest](https://github.com/google/googletest) for the test suites.
- [cpp-httplib](https://github.com/yhirose/cpp-httplib) by Yuji Hirose and [JSON for Modern C++](https://github.com/nlohmann/json) by Niels Lohmann, both MIT, for the playground server.

### Apple inference SDK (experimental)

Opt in with `-DTINYML_ENABLE_MPS=ON` and/or `-DTINYML_ENABLE_NPU=ON` on macOS. See [Apple inference SDK](docs/APPLE_INFERENCE.md) for backend selection, Core ML export, and Neural Engine fallback limitations.
