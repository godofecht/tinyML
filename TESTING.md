# Testing & CI Policy

## What runs in CI

CI runs on **pull requests to `main`**, **pushes to `main`** and **release tags** (`v*`).
Only fast unit tests execute in CI — the full suite finishes in under 2 minutes.

```
ctest --output-on-failure --timeout 120
```

## What does NOT run in CI (and why)

### Benchmarks (removed from ctest)

| Benchmark | Why skipped |
|---|---|
| `SIMDBenchmark` | Benchmark results on shared CI runners are meaningless — hardware varies per run |
| `AttentionBenchmark` | Same reason — timing-sensitive, needs dedicated hardware |
| `SimpleAttentionBenchmark` | Same reason |
| `ReinforcementLearningBenchmark` | Segfaults on Linux CI runners (works locally on macOS) |
| `GenerativeModelsBenchmark` | Benchmark — not a correctness test |

**Benchmarks are still built** so compilation is verified. They just aren't registered
with `add_test()` so `ctest` won't run them. Run them locally:

```bash
cd build
./bin/SIMDBenchmark
./bin/AttentionBenchmark
./bin/SimpleAttentionBenchmark
./bin/ReinforcementLearningBenchmark
./bin/GenerativeModelsBenchmark
```

### Timing assertions (skipped unless asked for)

Five tests assert wall-clock thresholds:

| Test | Asserts |
|---|---|
| `Phase1SIMDTest.PerformanceTargetsValidation` | per-op time against a target in µs |
| `Phase6ProductionTest.ProductionPerformanceBenchmarks` | audio, time series, vision and text latency |
| `Phase6ProductionTest.ProductionDeploymentScenarios` | speech, IoT, edge and device-text latency |
| `Phase4SimpleTest.StreamingSimulation` | jitter, as max/min per-token time |
| `Phase7AdvancedAttentionTest.PerformanceBenchmarks` | attention latency and throughput |

On a shared runner these measure the runner. The correctness assertions in the
same tests always run; the timing ones are opt-in:

```bash
TINYML_PERF_ASSERTS=1 ctest --output-on-failure
```

The measured numbers print either way. The current targets do not hold on a
GitHub runner, and `Phase7AdvancedAttentionTest` throughput sits near its 25
tok/s line even on a loaded laptop, so treat them as goals rather than facts.

### Disabled tests (registered but skipped)

| Test | Why disabled |
|---|---|
| `Phase8PhysicsInformedTest` | Takes **16+ minutes** on CI runners — too slow for free GitHub Actions |
| `Phase8PhysicsComprehensiveTest` | Non-deterministic convergence — `loss_ratio` swings from 0.04 to 21+ across runs |
| `Phase12ReinforcementTest` | Segfaults on Linux CI runners |

Run these locally if you need them:

```bash
cd build
ctest -R Phase8PhysicsInformedTest --force-new-ctest-process
ctest -R Phase12ReinforcementTest --force-new-ctest-process
```

## Releases

Pushing a version tag (e.g. `git tag v1.0.0 && git push --tags`) triggers:

1. Full build + unit tests
2. Release binary packaging (`libTinyML.a` + executables)
3. Upload to GitHub Releases

## Running the full suite locally

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Fast unit tests only (what CI runs)
ctest --output-on-failure --timeout 120

# Everything including disabled tests
ctest --output-on-failure --timeout 1200 --force-new-ctest-process
```
