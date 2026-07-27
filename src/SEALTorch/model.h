#pragma once

#include <cstddef>
#include <any>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sealtorch
{
    using TensorShape = std::vector<std::size_t>;

    struct DenseLayer
    {
        int input_size = 0;
        int output_size = 0;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
    };

    struct Convolution2D
    {
        std::vector<double> weights;
        std::vector<double> biases;
        std::size_t input_channels = 0;
        std::size_t output_channels = 0;
        std::size_t kernel_height = 0;
        std::size_t kernel_width = 0;
        std::size_t stride = 1;
        std::size_t padding = 0;
    };

    enum class PoolingType { Average, Maximum };

    struct Pooling2D
    {
        PoolingType type = PoolingType::Average;
        std::size_t height = 2;
        std::size_t width = 2;
        std::size_t stride = 2;
    };

    enum class ActivationType { Relu, Gelu };

    enum class OperationKind
    {
        Linear,
        Activation,
        Convolution2D,
        Pooling2D,
        Flatten,
        Custom
    };

    struct Operation
    {
        OperationKind kind = OperationKind::Flatten;
        std::string name = "Flatten";
        std::any payload;
        TensorShape input_shape;
        TensorShape output_shape;

        static Operation linear(DenseLayer layer, TensorShape input = {}, TensorShape output = {})
        {
            return {OperationKind::Linear, "Linear", std::move(layer), std::move(input), std::move(output)};
        }

        static Operation activation(ActivationType type)
        {
            return {OperationKind::Activation, "Activation", type, {}, {}};
        }

        static Operation convolution2d(Convolution2D layer, TensorShape input = {}, TensorShape output = {})
        {
            return {OperationKind::Convolution2D, "Convolution2D", std::move(layer), std::move(input), std::move(output)};
        }

        static Operation pooling2d(Pooling2D layer, TensorShape input = {}, TensorShape output = {})
        {
            return {OperationKind::Pooling2D, "Pooling2D", std::move(layer), std::move(input), std::move(output)};
        }

        static Operation flatten(TensorShape input = {}, TensorShape output = {})
        {
            return {OperationKind::Flatten, "Flatten", {}, std::move(input), std::move(output)};
        }

        template <typename Payload>
        static Operation custom(std::string operation_name, Payload payload,
                                TensorShape input = {}, TensorShape output = {})
        {
            return {OperationKind::Custom, std::move(operation_name),
                    std::any(std::move(payload)), std::move(input), std::move(output)};
        }
    };

    class Linear
    {
    public:
        Linear(int input_size, int output_size)
        {
            layer_.input_size = input_size;
            layer_.output_size = output_size;
            layer_.weights.assign(output_size, std::vector<double>(input_size, 0.0));
            layer_.biases.assign(output_size, 0.0);
        }

        explicit Linear(DenseLayer layer) : layer_(std::move(layer)) {}

        DenseLayer &parameters() { return layer_; }
        const DenseLayer &parameters() const { return layer_; }

    private:
        DenseLayer layer_;
    };

    class Activation
    {
    public:
        static Activation relu() { return Activation(ActivationType::Relu); }
        static Activation gelu() { return Activation(ActivationType::Gelu); }
        ActivationType type() const { return type_; }

    private:
        explicit Activation(ActivationType type) : type_(type) {}
        ActivationType type_;
    };

    class Sequential
    {
    public:
        Sequential() = default;

        // Generic extension point. The model remains an ordered pipeline;
        // operation-specific validation and execution belong to the backend.
        Sequential &add(Operation operation)
        {
            validate(operation);
            operations_.push_back(std::move(operation));
            dense_cache_valid_ = false;
            return *this;
        }

        Sequential &add(const Linear &layer)
        {
            DenseLayer dense = layer.parameters();
            TensorShape input = {static_cast<std::size_t>(dense.input_size)};
            TensorShape output = {static_cast<std::size_t>(dense.output_size)};
            for (auto reverse = operations_.rbegin(); reverse != operations_.rend(); ++reverse)
            {
                if (reverse->kind != OperationKind::Linear) continue;
                const auto &previous = std::any_cast<const DenseLayer &>(reverse->payload);
                if (previous.output_size != dense.input_size)
                    throw std::runtime_error("linear layer sizes do not match");
                break;
            }
            return add(Operation::linear(std::move(dense), std::move(input), std::move(output)));
        }

        Sequential &add(const Activation &activation)
        {
            if (operations_.empty()) throw std::runtime_error("activation needs a preceding layer");
            return add(Operation::activation(activation.type()));
        }

        const std::vector<Operation> &operations() const { return operations_; }

        // Convenience accessors for dense-only clients. They intentionally
        // reject mixed/CNN models instead of silently presenting bad data.
        const std::vector<DenseLayer> &layers() const
        {
            if (!dense_cache_valid_)
            {
                dense_cache_.clear();
                for (const Operation &operation : operations_)
                    if (operation.kind == OperationKind::Linear)
                        dense_cache_.push_back(std::any_cast<const DenseLayer &>(operation.payload));
                    else if (operation.kind != OperationKind::Activation)
                        throw std::runtime_error("layers() is only available for dense models");
                dense_cache_valid_ = true;
            }
            return dense_cache_;
        }

        int input_size() const { return first_linear_or_shape(true); }
        int output_size() const { return first_linear_or_shape(false); }

        bool has_activation(std::size_t index) const
        {
            const std::size_t operation_index = linear_operation_index(index);
            return operation_index + 1 < operations_.size() &&
                   operations_[operation_index + 1].kind == OperationKind::Activation;
        }

        ActivationType activation(std::size_t index) const
        {
            if (!has_activation(index)) throw std::runtime_error("operation has no activation");
            return std::any_cast<ActivationType>(operations_[linear_operation_index(index) + 1].payload);
        }

    private:
        void validate(const Operation &operation) const
        {
            if (operation.kind == OperationKind::Linear)
            {
                const auto &layer = std::any_cast<const DenseLayer &>(operation.payload);
                if (layer.input_size < 0 || layer.output_size < 0 ||
                    layer.weights.size() != static_cast<std::size_t>(layer.output_size) ||
                    layer.biases.size() != static_cast<std::size_t>(layer.output_size))
                    throw std::runtime_error("invalid dense layer dimensions");
                for (const auto &row : layer.weights)
                    if (row.size() != static_cast<std::size_t>(layer.input_size))
                        throw std::runtime_error("invalid dense layer weight dimensions");
            }
        }

        int first_linear_or_shape(bool input) const
        {
            if (!input)
            {
                for (auto reverse = operations_.rbegin(); reverse != operations_.rend(); ++reverse)
                {
                    if (reverse->kind == OperationKind::Linear)
                        return std::any_cast<const DenseLayer &>(reverse->payload).output_size;
                    if (!reverse->output_shape.empty()) return shape_size(reverse->output_shape);
                }
                return 0;
            }
            for (const Operation &operation : operations_)
            {
                if (operation.kind == OperationKind::Linear)
                    return std::any_cast<const DenseLayer &>(operation.payload).input_size;
                if (!operation.input_shape.empty()) return shape_size(operation.input_shape);
            }
            return 0;
        }

        static int shape_size(const TensorShape &shape)
        {
            std::size_t size = 1;
            for (std::size_t dimension : shape) size *= dimension;
            return static_cast<int>(size);
        }

        std::size_t linear_operation_index(std::size_t dense_index) const
        {
            std::size_t current = 0;
            for (std::size_t index = 0; index < operations_.size(); ++index)
                if (operations_[index].kind == OperationKind::Linear)
                {
                    if (current++ == dense_index) return index;
                }
            throw std::runtime_error("dense layer index is out of range");
        }

        std::vector<Operation> operations_;
        mutable bool dense_cache_valid_ = false;
        mutable std::vector<DenseLayer> dense_cache_;
    };

    using NeuralNetwork = Sequential;
}
