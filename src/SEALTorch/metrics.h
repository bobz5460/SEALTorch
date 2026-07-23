#pragma once

#include <cstddef>
#include <vector>

namespace sealtorch
{
    struct OutputComparison
    {
        double maximum_absolute_error = 0.0;
        double mean_absolute_error = 0.0;
        std::size_t different_values = 0;
    };

    OutputComparison compare_outputs(
        const std::vector<double> &expected,
        const std::vector<double> &actual,
        double tolerance = 1e-3);
}
