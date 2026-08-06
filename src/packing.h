#pragma once

#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sealtorch {

struct DiagonalSplit { std::size_t baby = 0; std::size_t giant = 0; };

inline std::vector<std::size_t> active_diagonals(const DenseLayer& layer, std::size_t slots) {
    std::vector<std::size_t> diagonals;
    std::vector<bool> active(slots, false);
    for (std::size_t row = 0; row < layer.weights.size(); ++row)
        for (std::size_t column = 0; column < layer.weights[row].size(); ++column)
            if (layer.weights[row][column] != 0.0)
                active[(column + slots - row) % slots] = true;
    for (std::size_t index = 0; index < slots; ++index)
        if (active[index]) diagonals.push_back(index);
    return diagonals;
}

inline std::size_t baby_step_size(std::size_t slots, std::size_t diagonal_count) {
    if (diagonal_count < 2) return 1;
    return std::min(slots, std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(diagonal_count))));
}

inline DiagonalSplit split_diagonal(std::size_t diagonal, std::size_t baby_step) {
    const std::size_t baby = diagonal % baby_step;
    return { baby, diagonal - baby };
}

inline std::int32_t signed_rotation(std::size_t rotation, std::size_t slots) {
    rotation %= slots;
    if (rotation > slots / 2) return static_cast<std::int32_t>(rotation) - static_cast<std::int32_t>(slots);
    return static_cast<std::int32_t>(rotation);
}

} // namespace sealtorch
