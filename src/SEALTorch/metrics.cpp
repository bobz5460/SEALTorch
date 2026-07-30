#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sealtorch
{
    OutputComparison compare_outputs(
        const std::vector<double> &expected,
        const std::vector<double> &actual,
        double tolerance)
    {
        if (expected.size() != actual.size())
            throw std::runtime_error("cannot compare outputs with different sizes");
        OutputComparison result;
        if (expected.empty()) return result;
        double total = 0.0;
        for (std::size_t index = 0; index < expected.size(); ++index)
        {
            const double error = std::abs(expected[index] - actual[index]);
            result.maximum_absolute_error = std::max(result.maximum_absolute_error, error);
            total += error;
            if (error > tolerance) ++result.different_values;
        }
        result.mean_absolute_error = total / expected.size();
        return result;
    }
}
