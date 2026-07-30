#pragma once

// Wall-clock thresholds on shared CI runners measure the runner, not the code.
// TESTING.md says this about the benchmark binaries; the same holds for the
// timing assertions embedded in these test files. They are skipped by default
// and enforced when you ask for them on hardware you control:
//
//     TINYML_PERF_ASSERTS=1 ctest --output-on-failure
//
// The measured numbers still print either way, so a run remains readable.

#include <cstdlib>
#include <cstring>

namespace tinyml_test {

inline bool perf_asserts_enabled() {
    const char* v = std::getenv("TINYML_PERF_ASSERTS");
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

}  // namespace tinyml_test

#define TINYML_IF_PERF_ASSERTS if (::tinyml_test::perf_asserts_enabled())
