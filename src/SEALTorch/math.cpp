#include "math.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace sealtorch
{
    namespace
    {
        double activation_value(ActivationType type, double value)
        {
            if (type == ActivationType::Relu)
                return std::max(0.0, value);
            if (type == ActivationType::Tanh)
                return std::tanh(value);
            return 0.5 * value *
                (1.0 + std::erf(value / std::sqrt(2.0)));
        }

        using PolynomialSystem = std::vector<std::vector<long double>>;

        std::vector<double> solve_system(
            PolynomialSystem system, std::size_t degree)
        {
            for (std::size_t column = 0; column <= degree; ++column) {
                std::size_t pivot = column;
                for (std::size_t row = column + 1; row <= degree; ++row)
                    if (std::abs(system[row][column]) >
                        std::abs(system[pivot][column]))
                        pivot = row;
                if (std::abs(system[pivot][column]) <
                    std::numeric_limits<long double>::epsilon() * 1e-6L)
                    throw std::runtime_error("could not fit activation polynomial");
                if (pivot != column)
                    std::swap(system[pivot], system[column]);
                const long double divisor = system[column][column];
                for (std::size_t item = column; item <= degree + 1; ++item)
                    system[column][item] /= divisor;
                for (std::size_t row = 0; row <= degree; ++row) {
                    if (row == column) continue;
                    const long double factor = system[row][column];
                    for (std::size_t item = column; item <= degree + 1; ++item)
                        system[row][item] -= factor * system[column][item];
                }
            }

            std::vector<double> coefficients(degree + 1);
            for (std::size_t order = 0; order <= degree; ++order) {
                coefficients[order] = static_cast<double>(
                    system[order][degree + 1]);
                if (std::abs(coefficients[order]) < 1e-12)
                    coefficients[order] = 0.0;
            }
            while (coefficients.size() > 2 && coefficients.back() == 0.0)
                coefficients.pop_back();
            return coefficients;
        }

        std::vector<double> taylor_coefficients(
            ActivationType type, std::size_t degree)
        {
            std::vector<double> coefficients;
            if (type == ActivationType::Relu)
                coefficients = {0.0, 1.0};
            else if (type == ActivationType::Tanh)
                coefficients = {0.0, 1.0, 0.0, -1.0 / 3.0};
            else
                coefficients = {
                    0.0, 0.5,
                    1.0 / std::sqrt(2.0 * std::numbers::pi), 0.0,
                    -1.0 / (6.0 * std::sqrt(2.0 * std::numbers::pi))};
            coefficients.resize(std::min(degree + 1, coefficients.size()));
            while (coefficients.size() > 2 && coefficients.back() == 0.0)
                coefficients.pop_back();
            return coefficients;
        }

        bool is_odd_polynomial(const std::vector<double> &coefficients)
        {
            if (coefficients.size() < 4) return false;
            for (std::size_t order = 0; order < coefficients.size(); order += 2)
                if (coefficients[order] != 0.0) return false;
            return true;
        }

        void add_plain_scalar(
            const seal::Evaluator &evaluator, seal::CKKSEncoder &encoder,
            seal::Ciphertext &value, double coefficient)
        {
            if (coefficient == 0.0) return;
            seal::Plaintext plain;
            encoder.encode(coefficient, value.scale(), plain);
            evaluator.mod_switch_to_inplace(plain, value.parms_id());
            evaluator.add_plain_inplace(value, plain);
        }
    }

    std::vector<double> activation_polynomial_coefficients(
        ActivationType type, std::size_t degree, double range,
        ActivationApproximation method)
    {
        if (degree < 1 || degree > 15)
            throw std::invalid_argument("activation degree must be in [1, 15]");
        if (!std::isfinite(range) || range <= 0.0 || range > 32.0)
            throw std::invalid_argument("activation range must be in (0, 32]");

        if (method == ActivationApproximation::Taylor)
            return taylor_coefficients(type, degree);

        PolynomialSystem system(
            degree + 1, std::vector<long double>(degree + 2, 0.0L));
        if (method == ActivationApproximation::Chebyshev) {
            // Interpolate at first-kind Chebyshev roots, then solve for the
            // equivalent power-basis coefficients used by both HE backends.
            const std::size_t count = degree + 1;
            for (std::size_t row = 0; row < count; ++row) {
                const double x = range * std::cos(
                    std::numbers::pi * (2.0 * static_cast<double>(row) + 1.0) /
                    (2.0 * static_cast<double>(count)));
                long double power = 1.0L;
                for (std::size_t column = 0; column <= degree; ++column) {
                    system[row][column] = power;
                    power *= static_cast<long double>(x);
                }
                system[row][degree + 1] = activation_value(type, x);
            }
        } else {
            // Uniform-grid, unweighted L2 least-squares fit.
            constexpr std::size_t sample_count = 257;
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                const double x = -range + 2.0 * range *
                    static_cast<double>(sample) /
                    static_cast<double>(sample_count - 1);
                const double target = activation_value(type, x);
                std::vector<long double> powers(2 * degree + 1, 0.0L);
                powers[0] = 1.0;
                for (std::size_t order = 1; order <= 2 * degree; ++order)
                    powers[order] = powers[order - 1] *
                        static_cast<long double>(x);
                for (std::size_t row = 0; row <= degree; ++row) {
                    for (std::size_t column = 0; column <= degree; ++column)
                        system[row][column] += powers[row + column];
                    system[row][degree + 1] += target * powers[row];
                }
            }
        }
        return solve_system(system, degree);
    }

    std::size_t activation_polynomial_depth(
        ActivationType type, std::size_t degree, double range,
        ActivationApproximation method)
    {
        const auto coefficients = activation_polynomial_coefficients(
            type, degree, range, method);
        const std::size_t highest = coefficients.size() - 1;
        return is_odd_polynomial(coefficients)
            ? (highest + 3) / 2
            : highest;
    }

    seal::Ciphertext approximate_activation(
        const seal::Evaluator &evaluator, const seal::RelinKeys &relin_keys,
        seal::CKKSEncoder &encoder, const seal::Ciphertext &input, double scale,
        ActivationType type, std::size_t degree, double range,
        ActivationApproximation method)
    {
        const std::vector<double> coefficients =
            activation_polynomial_coefficients(type, degree, range, method);
        const std::size_t highest = coefficients.size() - 1;

        if (is_odd_polynomial(coefficients)) {
            seal::Ciphertext square;
            evaluator.square(input, square);
            evaluator.relinearize_inplace(square, relin_keys);
            evaluator.rescale_to_next_inplace(square);

            seal::Ciphertext result = square;
            seal::Plaintext leading;
            encoder.encode(coefficients[highest], scale, leading);
            evaluator.mod_switch_to_inplace(leading, result.parms_id());
            evaluator.multiply_plain_inplace(result, leading);
            evaluator.rescale_to_next_inplace(result);

            for (std::size_t order = highest - 2; order > 1; order -= 2) {
                add_plain_scalar(evaluator, encoder, result, coefficients[order]);
                seal::Ciphertext factor = square;
                evaluator.mod_switch_to_inplace(factor, result.parms_id());
                evaluator.multiply_inplace(result, factor);
                evaluator.relinearize_inplace(result, relin_keys);
                evaluator.rescale_to_next_inplace(result);
            }
            add_plain_scalar(evaluator, encoder, result, coefficients[1]);
            seal::Ciphertext factor = input;
            evaluator.mod_switch_to_inplace(factor, result.parms_id());
            evaluator.multiply_inplace(result, factor);
            evaluator.relinearize_inplace(result, relin_keys);
            evaluator.rescale_to_next_inplace(result);
            return result;
        }

        seal::Ciphertext result = input;
        if (coefficients[highest] != 1.0) {
            seal::Plaintext coefficient;
            encoder.encode(coefficients[highest], scale, coefficient);
            evaluator.mod_switch_to_inplace(coefficient, result.parms_id());
            evaluator.multiply_plain_inplace(result, coefficient);
            evaluator.rescale_to_next_inplace(result);
        }

        for (std::size_t order = highest; order-- > 1;) {
            if (coefficients[order] != 0.0) {
                seal::Plaintext coefficient;
                encoder.encode(coefficients[order], result.scale(), coefficient);
                evaluator.mod_switch_to_inplace(coefficient, result.parms_id());
                evaluator.add_plain_inplace(result, coefficient);
            }
            seal::Ciphertext factor = input;
            evaluator.mod_switch_to_inplace(factor, result.parms_id());
            evaluator.multiply_inplace(result, factor);
            evaluator.relinearize_inplace(result, relin_keys);
            evaluator.rescale_to_next_inplace(result);
        }
        if (coefficients[0] != 0.0) {
            seal::Plaintext constant;
            encoder.encode(coefficients[0], result.scale(), constant);
            evaluator.mod_switch_to_inplace(constant, result.parms_id());
            evaluator.add_plain_inplace(result, constant);
        }
        return result;
    }

    seal::Ciphertext encrypted_matrix_vector_product(
        const seal::SEALContext& context,
        const seal::Evaluator& evaluator,
        const seal::GaloisKeys& galois_keys,
        const seal::Ciphertext& input,
        const EncodedMatrix& matrix,
        std::size_t thread_count,
        ThreadPool &thread_pool)
    {
        // Baby-step/giant-step evaluates many diagonals while using only about
        // 2*sqrt(slot_count) rotations. Rotation keys are much larger than
        // plaintext weights, so this substantially reduces memory use.
        std::vector<seal::Ciphertext> rotated_inputs(
            matrix.input_rotations.size());
        thread_pool.parallel_for_workers(
            matrix.input_rotations.size(),
            thread_count,
            [&](std::size_t worker, std::size_t jobs)
            {
                seal::Evaluator local_evaluator(context);
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();
                for (std::size_t index = worker;
                     index < matrix.input_rotations.size();
                     index += jobs)
                {
                    const int rotation = matrix.input_rotations[index];
                    if (rotation == 0)
                        rotated_inputs[index] = input;
                    else
                        local_evaluator.rotate_vector(
                            input,
                            rotation,
                            galois_keys,
                            rotated_inputs[index],
                            pool);
                }
            });

        std::vector<seal::Ciphertext> group_results(matrix.groups.size());
        thread_pool.parallel_for_workers(
            matrix.groups.size(),
            thread_count,
            [&](std::size_t worker, std::size_t jobs)
            {
                seal::Evaluator local_evaluator(context);
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();
                for (std::size_t group_index = worker;
                     group_index < matrix.groups.size();
                     group_index += jobs)
                {
                    const EncodedDiagonalGroup &group =
                        matrix.groups[group_index];
                    seal::Ciphertext subtotal;
                    bool first = true;
                    for (const EncodedDiagonal &diagonal : group.diagonals)
                    {
                        seal::Ciphertext term;
                        local_evaluator.multiply_plain(
                            rotated_inputs[diagonal.input_index],
                            diagonal.weights,
                            term,
                            pool);
                        if (first) {
                            subtotal = std::move(term);
                            first = false;
                        } else {
                            local_evaluator.add_inplace(subtotal, term);
                        }
                    }

                    if (group.rotation == 0)
                        group_results[group_index] = std::move(subtotal);
                    else
                        local_evaluator.rotate_vector(
                            subtotal,
                            group.rotation,
                            galois_keys,
                            group_results[group_index],
                            pool);
                }
            });

        seal::Ciphertext result = std::move(group_results.front());
        for (std::size_t index = 1; index < group_results.size(); ++index)
        {
            evaluator.add_inplace(result, group_results[index]);
        }
        return result;
    }

}
