# Testing & CI Policy

## Release gates

CI runs on pull requests to `main`, pushes to `main` and version tags. The build matrix contains a dependency-free core configuration plus full GCC and Clang configurations.

Every matrix entry performs four distinct checks: configure/build, CTest correctness tests, `cmake --install`, and a fresh downstream consumer build using `find_package(TinyML CONFIG REQUIRED COMPONENTS Core)`. This catches packaging/export errors that an in-tree build cannot detect.

Core-only verification:

```bash
cmake -S . -B build-core -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTINYML_BUILD_EXTENDED=OFF
cmake --build build-core --parallel
ctest --test-dir build-core --output-on-failure
```

Full verification:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure --timeout 120
```

Installed-consumer verification:

```bash
cmake --install build --prefix "$PWD/install"
cmake -S tests/install_consumer -B consumer-build -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build consumer-build
ctest --test-dir build --output-on-failure
./consumer-build/tinyml_install_consumer
```

## Performance assertions

Wall-clock thresholds are not release correctness gates on shared CI hardware. The tests still print measured timings, while assertions are enabled locally with:

```bash
TINYML_PERF_ASSERTS=1 ctest --test-dir build --output-on-failure
```

## Benchmarks

Benchmarks are excluded from normal builds and from CTest. Enable the supported benchmark targets explicitly:

```bash
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DTINYML_BUILD_TESTS=OFF \
  -DTINYML_BUILD_BENCHMARKS=ON
cmake --build build-bench --parallel
```

Benchmark numbers should be recorded with compiler, flags, CPU, operating system, power mode and dataset/input shape. Results from an unspecified shared runner are not suitable for performance claims.

## Registered disabled tests

The following tests remain compiled but disabled in the default CTest run:

| Test | Status |
| --- | --- |
| `Phase8PhysicsInformedTest` | Long-running experiment; unsuitable for the fast release gate |
| `Phase8PhysicsComprehensiveTest` | Non-deterministic convergence; requires a deterministic acceptance criterion |
| `Phase12ReinforcementTest` | Known Linux failure; module remains preview until fixed |

A module with a known-failure test cannot graduate to the stable API surface.

Run an individual disabled test locally with CTest's disabled-test override, for example:

```bash
ctest --test-dir build -R Phase12ReinforcementTest \
  --output-on-failure --force-new-ctest-process
```

## Sanitizers and dedicated performance hardware

Sanitizer runs and dedicated-hardware performance baselines are appropriate release-hardening checks, but they are intentionally separate from the current fast CI matrix. They should be added only when the corresponding test corpus is deterministic enough that a failure represents a code defect rather than runner variance.

## Releases

A `v*` tag is packaged from the CMake install graph. The release job produces separate core and extended archives and SHA-256 checksums. The core archive contains no xsimd dependency. The extended package records xsimd as a CMake dependency rather than vendoring it into the TinyML SDK.
