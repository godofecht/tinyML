// SPDX-License-Identifier: MIT
#pragma once

#include <Model.h>
#include <functional>
#include <string>
#include <vector>

namespace ML
{
    enum class Backend { CPU, MPS, NPU };

    // Inference-only snapshot. Recreate after training or changing weights.
    // Sessions are synchronous and not safe for concurrent calls.
    class InferenceSession
    {
    public:
        explicit InferenceSession (const Model& model, Backend backend = Backend::CPU);
        InferenceSession (InferenceSession&&) noexcept = default;
        InferenceSession& operator= (InferenceSession&&) noexcept = default;
        InferenceSession (const InferenceSession&) = delete;
        InferenceSession& operator= (const InferenceSession&) = delete;
        // Loads an independently exported Core ML artifact; NPU permits CPU execution.
        static InferenceSession loadNPU (const std::string& compiledModelPath);
        // CPU/MPS availability; for NPU indicates SDK support, not physical ANE use.
        static bool available (Backend backend) noexcept;

        Backend backend() const noexcept { return selectedBackend; }
        bool mayUseCPU() const noexcept { return selectedBackend != Backend::MPS; }
        std::vector<double> predict (const std::vector<double>& input);

    private:
        using Runner = std::function<std::vector<double> (const std::vector<double>&)>;
        InferenceSession (Backend backend, Runner runner);
        Backend selectedBackend;
        Runner run;
    };
}
