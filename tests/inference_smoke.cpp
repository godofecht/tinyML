#include <tinyml/core.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main (int argc, char** argv)
{
    ML::Model model ({2, 3, 2});
    model.setWeights ({0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.07, -0.08, 0.09,
                       0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.15, 0.12});
    if (argc == 2 && std::string(argv[1]) == "--export-fixture")
    {
        model.saveWeightsToFile ("apple-fixture.txt");
        return 0;
    }
    ML::InferenceSession cpu (model);
    const std::vector<double> input {0.25, -0.5};
    model.feedForward (input);
    const auto reference = model.getResult();
    auto check = [&] (ML::InferenceSession& session, double tolerance)
    {
        for (const auto& sample : std::vector<std::vector<double>> {input, {-0.4, 0.8}, {0.0, 0.0}})
        {
            model.feedForward (sample);
            const auto expected = model.getResult();
            const auto actual = session.predict (sample);
            if (actual.size() != expected.size()) return false;
            for (std::size_t i = 0; i < actual.size(); ++i)
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance)
                    return false;
        }
        try { session.predict ({1.0}); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };
    if (!check(cpu, 1e-12) || cpu.backend() != ML::Backend::CPU) return 1;
    if (argc == 2 && std::string(argv[1]) == "--mps")
    {
        if (!ML::InferenceSession::available (ML::Backend::MPS))
        {
            std::cout << "No Metal device available\n";
            return 77;
        }
        ML::InferenceSession gpu (model, ML::Backend::MPS);
        return check(gpu, 2e-5) && !gpu.mayUseCPU() ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--npu")
    {
        auto npu = ML::InferenceSession::loadNPU (argv[2]);
        return check(npu, 3e-3) && npu.mayUseCPU() ? 0 : 1;
    }
    if (!ML::InferenceSession::available (ML::Backend::MPS))
    {
        try { ML::InferenceSession unsupported (model, ML::Backend::MPS); return 1; }
        catch (const std::runtime_error&) {}
    }
    model.setWeights (std::vector<double> (model.getWeights().size(), 0.0));
    return cpu.predict(input) == reference ? 0 : 1;
}
