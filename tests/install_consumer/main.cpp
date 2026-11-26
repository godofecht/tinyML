#include <tinyml/core.hpp>
#include <cstdlib>

int main()
{
    ML::Network network ({ 2, 2, 1 });
    network.feedForward ({ 0.0, 1.0 });
    std::vector<double> result;
    network.getResults (result);
    ML::Model model ({2, 2, 1});
    ML::InferenceSession session (model);
    return result.size() == 1 && session.predict ({0.0, 1.0}).size() == 1
        ? EXIT_SUCCESS : EXIT_FAILURE;
}
