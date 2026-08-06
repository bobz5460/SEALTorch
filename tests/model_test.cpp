#include <SEALTorch/model.h>
#include <SEALTorch/math.h>
#include <SEALTorch/packing.h>

#include <cassert>
#include <cmath>
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
        assert(sealtorch::baby_step_size(8, diagonals.size()) == 2);

        const sealtorch::DiagonalSplit split =
            sealtorch::split_diagonal(7, 2);
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
        model.add(small_layer());
        model.add(sealtorch::ActivationType::Relu);
        assert(model.input_size() == 3);
        assert(model.output_size() == 2);
        assert(model.operations().size() == 2);
    }

    double polynomial(const std::vector<double> &coefficients, double value)
    {
        double result = 0.0;
        for (auto item = coefficients.rbegin(); item != coefficients.rend(); ++item)
            result = result * value + *item;
        return result;
    }

    void test_interval_activation_polynomials()
    {
        const auto tanh = sealtorch::activation_polynomial_coefficients(
            sealtorch::ActivationType::Tanh, 3, 4.0);
        assert(tanh.size() == 4);
        assert(std::abs(polynomial(tanh, 4.0) - std::tanh(4.0)) < 0.4);
        assert(std::abs(polynomial(tanh, -4.0) - std::tanh(-4.0)) < 0.4);

        const auto relu = sealtorch::activation_polynomial_coefficients(
            sealtorch::ActivationType::Relu, 4, 4.0);
        assert(relu.size() == 5);
        assert(std::abs(polynomial(relu, 2.0) - 2.0) < 0.25);

        const auto narrow = sealtorch::activation_polynomial_coefficients(
            sealtorch::ActivationType::Gelu, 4, 2.0);
        const auto wide = sealtorch::activation_polynomial_coefficients(
            sealtorch::ActivationType::Gelu, 4, 4.0);
        assert(narrow != wide);

        const auto taylor = sealtorch::activation_polynomial_coefficients(
            sealtorch::ActivationType::Tanh, 3, 4.0,
            sealtorch::ActivationApproximation::Taylor);
        assert((taylor == std::vector<double>{
            0.0, 1.0, 0.0, -1.0 / 3.0}));

        const auto chebyshev = sealtorch::activation_polynomial_coefficients(
            sealtorch::ActivationType::Tanh, 3, 4.0,
            sealtorch::ActivationApproximation::Chebyshev);
        assert(chebyshev != tanh);
        assert(std::abs(polynomial(chebyshev, 4.0) - std::tanh(4.0)) < 0.5);

        const auto degree_seven =
            sealtorch::activation_polynomial_coefficients(
                sealtorch::ActivationType::Tanh, 7, 6.0,
                sealtorch::ActivationApproximation::Chebyshev);
        assert(degree_seven.size() == 8);
        assert(std::abs(polynomial(degree_seven, 6.0) - std::tanh(6.0)) < 0.25);
        assert(sealtorch::activation_polynomial_depth(
            sealtorch::ActivationType::Tanh, 5, 3.5,
            sealtorch::ActivationApproximation::Chebyshev) == 4);
        assert(sealtorch::activation_polynomial_depth(
            sealtorch::ActivationType::Tanh, 7, 6.0,
            sealtorch::ActivationApproximation::Chebyshev) == 5);
    }
}

int main()
{
    test_sparse_diagonals();
    test_zero_layer();
    test_sequential_model();
    test_interval_activation_polynomials();
}
