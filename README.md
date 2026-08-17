# TinyML

[![CI](https://github.com/godofecht/tinyML/actions/workflows/ci.yml/badge.svg)](https://github.com/godofecht/tinyML/actions/workflows/ci.yml)
[![Pages](https://github.com/godofecht/tinyML/actions/workflows/pages.yml/badge.svg)](https://godofecht.github.io/tinyML/)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20core%20%2B%20commercial-blue.svg)

**TinyML is a lightweight, high-performance C++ machine-learning and statistical-computing library aimed at real-time and embedded use.** It implements the stack directly in modern C++, with SIMD-aware kernels, quantization, neural-network architectures, scientific ML, reinforcement learning, graph models, generative models and an interactive playground.

**Project site:** https://godofecht.github.io/tinyML/

## What is here

The library includes feed-forward networks, CNNs, RNNs, streaming Transformers, Bayesian neural networks, VAEs, GANs, PINNs, policy-gradient and Q-learning examples, graph neural networks, time-series forecasting, quantized inference, SIMD operations and production-oriented serving helpers.

The `playground/` directory contains a C++ backend plus a browser frontend for interactive scenarios such as CartPole, Pong, attention visualisation, CNN operations, PINNs and graph diffusion. The GitHub Pages site is a separate static project front page; the full playground still runs against the local C++ server.

## Build

```bash
git clone https://github.com/godofecht/tinyML.git
cd tinyML
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CMake fetches xsimd and GoogleTest when they are not already available. See [TESTING.md](TESTING.md) for the distinction between correctness tests, disabled long-running tests and opt-in timing assertions.

## Playground

```bash
cmake --build build --target PlaygroundServer --parallel
./build/bin/PlaygroundServer
```

Then open `http://localhost:8081`.

## Repository map

```text
include/       public library headers
src/           implementation
examples/      usage examples
benchmarks/    local benchmark programs
playground/    C++ server + browser UI
tests/         GoogleTest suite
docs/          API, demo and wiki documentation
blog/          implementation and architecture notes
site/          static GitHub Pages site
```

## CI and releases

Every pull request and push to `main` builds the project with both GCC and Clang and runs the registered CTest suite. Version tags matching `v*` additionally produce a Linux x86-64 release archive.

Wall-clock performance assertions are intentionally opt-in because shared CI hardware is not a meaningful benchmark environment. Use `TINYML_PERF_ASSERTS=1` locally when you explicitly want those thresholds enforced.

## Documentation

Start with [PRODUCTION_API.md](docs/PRODUCTION_API.md), [TESTING.md](TESTING.md), [ROADMAP.md](ROADMAP.md), [DEMO_ROADMAP.md](docs/DEMO_ROADMAP.md) and the material in `docs/wiki/` and `blog/`.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Security-sensitive reports should follow [SECURITY.md](SECURITY.md).

## Licensing

TinyML is dual licensed. The zero-dependency core is MIT-licensed, including commercial use. The extended model library is covered by the commercial terms described in [LICENSING.md](LICENSING.md). That file is the authoritative map of which headers belong to each side of the license boundary.
