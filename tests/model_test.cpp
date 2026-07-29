#include <SEALTorch/model.h>
#include <SEALTorch/packing.h>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace
{
    sealtorch::DenseLayer small_layer()
    {
        sealtorch::DenseLayer layer;
        layer.input_size = 3;
        layer.output_size = 2;
        layer.weights = {
            {1.0, 0.0, 0.0},
            {0.0, 0.0, 2.0},
        };
        layer.biases = {0.0, 0.0};
        return layer;
    }

    void test_sparse_diagonals()
    {
        const std::vector<std::size_t> diagonals =
            sealtorch::active_diagonals(small_layer(), 8);
        assert((diagonals == std::vector<std::size_t>{0, 1}));
        assert(sealtorch::signed_rotation(0, 8) == 0);
        assert(sealtorch::signed_rotation(7, 8) == -1);
        assert(sealtorch::baby_step_size(8) == 3);

        const sealtorch::DiagonalSplit split =
            sealtorch::split_diagonal(7, 8);
        assert(split.baby == 1);
        assert(split.giant == 6);
        assert((split.baby + split.giant) % 8 == 7);
    }

    void test_zero_layer()
    {
        sealtorch::DenseLayer layer = small_layer();
        layer.weights.assign(2, std::vector<double>(3, 0.0));
        const std::vector<std::size_t> diagonals =
            sealtorch::active_diagonals(layer, 8);
        assert((diagonals == std::vector<std::size_t>{0}));
    }

    void test_sequential_model()
    {
        sealtorch::Sequential model;
        model.add(sealtorch::Linear(small_layer()));
        model.add(sealtorch::Activation::relu());
        assert(model.input_size() == 3);
        assert(model.output_size() == 2);
        assert(model.has_activation(0));
        assert(model.activation(0) == sealtorch::ActivationType::Relu);
    }

    void test_invalid_dimensions()
    {
        sealtorch::DenseLayer layer = small_layer();
        layer.biases.pop_back();
        bool rejected = false;
        try {
            sealtorch::Sequential().add(sealtorch::Linear(layer));
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        assert(rejected);
    }
}

int main()
{
    test_sparse_diagonals();
    test_zero_layer();
    test_sequential_model();
    test_invalid_dimensions();
}
