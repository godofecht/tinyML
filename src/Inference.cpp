// SPDX-License-Identifier: MIT
#include <tinyml/inference.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ML
{
    namespace detail
    {
#if TINYML_ENABLE_MPS
        bool mpsAvailable() noexcept;
        std::function<std::vector<double> (const std::vector<double>&)> makeMPS (const Model&);
#endif
#if TINYML_ENABLE_NPU
        bool coreMLAvailable() noexcept;
        std::function<std::vector<double> (const std::vector<double>&)> makeNPU (const std::string&);
#endif
    }

    InferenceSession::InferenceSession (Backend backend, Runner runner)
        : selectedBackend (backend), run (std::move (runner))
    {
    }

    InferenceSession::InferenceSession (const Model& model, Backend backend)
        : selectedBackend (backend)
    {
        switch (backend)
        {
            case Backend::CPU:
            {
                auto snapshot = std::make_shared<Model> (model.getTopology());
                snapshot->setWeights (model.getWeights());
                run = [snapshot] (const std::vector<double>& input)
                {
                    snapshot->feedForward (input);
                    return snapshot->getResult();
                };
                break;
            }
            case Backend::MPS:
#if TINYML_ENABLE_MPS
                run = detail::makeMPS (model);
                break;
#else
                throw std::runtime_error ("TinyML MPS backend was not enabled at build time");
#endif
            case Backend::NPU:
                throw std::invalid_argument ("Use InferenceSession::loadNPU with an exported .mlmodelc");
            default:
                throw std::invalid_argument ("Unknown TinyML backend");
        }
    }

    InferenceSession InferenceSession::loadNPU (const std::string& compiledModelPath)
    {
#if TINYML_ENABLE_NPU
        return InferenceSession (Backend::NPU, detail::makeNPU (compiledModelPath));
#else
        (void) compiledModelPath;
        throw std::runtime_error ("TinyML NPU backend was not enabled at build time");
#endif
    }

    bool InferenceSession::available (Backend backend) noexcept
    {
        switch (backend)
        {
            case Backend::CPU: return true;
            case Backend::MPS:
#if TINYML_ENABLE_MPS
                return detail::mpsAvailable();
#else
                return false;
#endif
            case Backend::NPU:
#if TINYML_ENABLE_NPU
                return detail::coreMLAvailable();
#else
                return false;
#endif
            default: return false;
        }
    }

    std::vector<double> InferenceSession::predict (const std::vector<double>& input)
    {
        return run (input);
    }
}
