#pragma once

#include <SEALTorch/model.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sealtorch
{
    // Return only diagonals that contain at least one non-zero weight.
    // Keeping this list sparse avoids unnecessary rotations, keys, and
    // encoded plaintexts for convolution and pooling matrices.
    inline std::vector<std::size_t> active_diagonals(
        const DenseLayer &layer,
        std::size_t slot_count)
    {
        std::vector<bool> used(slot_count, false);

        for (std::size_t row = 0; row < layer.weights.size(); ++row)
        {
            for (std::size_t column = 0; column < layer.weights[row].size(); ++column)
            {
                if (layer.weights[row][column] != 0.0)
                    used[(column + slot_count - row) % slot_count] = true;
            }
        }

        std::vector<std::size_t> diagonals;
        for (std::size_t diagonal = 0; diagonal < slot_count; ++diagonal)
        {
            if (used[diagonal])
                diagonals.push_back(diagonal);
        }

        // A zero-weight layer still needs a valid ciphertext before its bias
        // can be added. Multiplying diagonal zero by an all-zero plaintext is
        // the cheapest way to create one.
        if (diagonals.empty())
            diagonals.push_back(0);

        return diagonals;
    }

    inline std::int32_t signed_rotation(
        std::size_t diagonal,
        std::size_t slot_count)
    {
        if (diagonal <= slot_count / 2)
            return static_cast<std::int32_t>(diagonal);
        return static_cast<std::int32_t>(diagonal) -
               static_cast<std::int32_t>(slot_count);
    }

    inline std::size_t baby_step_size(std::size_t slot_count)
    {
        std::size_t size = 1;
        while (size * size < slot_count)
            ++size;
        return size;
    }

    struct DiagonalSplit
    {
        std::size_t baby = 0;
        std::size_t giant = 0;
    };

    inline DiagonalSplit split_diagonal(
        std::size_t diagonal,
        std::size_t slot_count)
    {
        const std::size_t baby_step = baby_step_size(slot_count);
        return {
            diagonal % baby_step,
            diagonal - diagonal % baby_step,
        };
    }
}
