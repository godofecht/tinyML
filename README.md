# tinyML

[![CI](https://github.com/godofecht/tinyML/actions/workflows/ci.yml/badge.svg)](https://github.com/godofecht/tinyML/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/godofecht/tinyML/actions/workflows/pages.yml/badge.svg)](https://github.com/godofecht/tinyML/actions/workflows/pages.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17%2B-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

A lightweight C++ machine-learning library for embedded, edge and real-time workloads.

The repository contains two intentionally different surfaces. `TinyML::Core` is the supported, dependency-free library surface with an install/export contract and semantic-versioning guarantees. `TinyML::Extended` contains the xsimd-backed model stack and research modules. The extended target is built by default for source compatibility, but its individual modules have their own stability status documented in [docs/STABILITY.md](docs/STABILITY.md).

## Requirements

The core requires a C++17 compiler and CMake 3.20 or newer. It has no third-party runtime or build dependency.

The extended target additionally requires xsimd 12.1 or newer. Top-level source builds can fetch xsimd automatically; package consumers are expected to provide it through their normal dependency manager.

## Build

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

Benchmarks, examples and the playground are deliberately excluded from ordinary library builds. Enable them explicitly with `TINYML_BUILD_BENCHMARKS`, `TINYML_BUILD_EXAMPLES` or `TINYML_BUILD_PLAYGROUND`.

## Install and consume

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

## Build options

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

## Testing and release guarantees

CI builds the zero-dependency core separately from the full library with GCC and Clang. Every configuration is installed into a staging prefix and then consumed by a fresh downstream CMake project through `find_package`, so packaging regressions fail before release.

Timing assertions are opt-in because shared CI hardware is unsuitable for performance gates. Benchmark executables are local-only. See [TESTING.md](TESTING.md).

Versioned releases are assembled from `cmake --install`, not by manually copying build artifacts. Core and extended archives are published separately, with SHA-256 checksums.

## API stability

The compatibility contract is explicit rather than implied by the presence of a header in the repository. See [docs/STABILITY.md](docs/STABILITY.md) for the stable, preview and source-only surfaces.

Changes to the stable core follow semantic versioning. Preview and source-only modules may change before they graduate into the stable surface.

## Licensing

The installable core is MIT licensed. The extended model library contains commercially licensed modules and xsimd-backed functionality. Individual file licensing and the distinction between licensing and API stability are documented in [LICENSING.md](LICENSING.md). Commercial terms are in [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

Security reports should follow [SECURITY.md](SECURITY.md). Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md). Release history is tracked in [CHANGELOG.md](CHANGELOG.md).
