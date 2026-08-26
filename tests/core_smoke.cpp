#include <tinyml/core.hpp>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

int main()
{
    ML::Model model ({ 2, 3, 1 });
    model.feedForward ({ 0.25, -0.5 });

    const auto result = model.getResult();
    if (result.size() != 1 || !std::isfinite (result.front()))
    {
        return EXIT_FAILURE;
    }

    const auto weights = model.getWeights();
    if (weights.empty())
    {
        return EXIT_FAILURE;
    }

    model.setWeights (weights);

    bool rejectedBadInput = false;
    try
    {
        model.feedForward ({ 1.0 });
    }
    catch (const std::invalid_argument&)
    {
        rejectedBadInput = true;
    }

    if (!rejectedBadInput)
    {
        return EXIT_FAILURE;
    }

    bool rejectedBadWeights = false;
    try
    {
        model.setWeights ({ 1.0 });
    }
    catch (const std::invalid_argument&)
    {
        rejectedBadWeights = true;
    }

    return rejectedBadWeights ? EXIT_SUCCESS : EXIT_FAILURE;
}
