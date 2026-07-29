#pragma once

#include <cstddef>
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
        TensorShape input_shape;
        TensorShape output_shape;

        // An operation uses the field that matches its kind.
        DenseLayer linear_layer;
        ActivationType activation_type = ActivationType::Relu;
        Convolution2D convolution_layer;
        Pooling2D pooling_layer;

        static Operation linear(DenseLayer layer, TensorShape input = {}, TensorShape output = {})
        {
            Operation operation;
            operation.kind = OperationKind::Linear;
            operation.name = "Linear";
            operation.linear_layer = std::move(layer);
            operation.input_shape = std::move(input);
            operation.output_shape = std::move(output);
            return operation;
        }

        static Operation activation(ActivationType type)
        {
            Operation operation;
            operation.kind = OperationKind::Activation;
            operation.name = "Activation";
            operation.activation_type = type;
            return operation;
        }

        static Operation convolution2d(Convolution2D layer, TensorShape input = {}, TensorShape output = {})
        {
            Operation operation;
            operation.kind = OperationKind::Convolution2D;
            operation.name = "Convolution2D";
            operation.convolution_layer = std::move(layer);
            operation.input_shape = std::move(input);
            operation.output_shape = std::move(output);
            return operation;
        }

        static Operation pooling2d(Pooling2D layer, TensorShape input = {}, TensorShape output = {})
        {
            Operation operation;
            operation.kind = OperationKind::Pooling2D;
            operation.name = "Pooling2D";
            operation.pooling_layer = std::move(layer);
            operation.input_shape = std::move(input);
            operation.output_shape = std::move(output);
            return operation;
        }

        static Operation flatten(TensorShape input = {}, TensorShape output = {})
        {
            Operation operation;
            operation.kind = OperationKind::Flatten;
            operation.name = "Flatten";
            operation.input_shape = std::move(input);
            operation.output_shape = std::move(output);
            return operation;
        }

        static Operation custom(std::string operation_name, TensorShape input = {}, TensorShape output = {})
        {
            Operation operation;
            operation.kind = OperationKind::Custom;
            operation.name = std::move(operation_name);
            operation.input_shape = std::move(input);
            operation.output_shape = std::move(output);
            return operation;
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
            for (std::size_t index = operations_.size(); index > 0; --index)
            {
                const Operation &previous_operation = operations_[index - 1];
                if (previous_operation.kind != OperationKind::Linear) continue;
                const DenseLayer &previous = previous_operation.linear_layer;
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
                        dense_cache_.push_back(operation.linear_layer);
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
            return operations_[linear_operation_index(index) + 1].activation_type;
        }

    private:
        void validate(const Operation &operation) const
        {
            if (operation.kind == OperationKind::Linear)
            {
                const DenseLayer &layer = operation.linear_layer;
                if (layer.input_size < 0 || layer.output_size < 0 ||
                    layer.weights.size() != static_cast<std::size_t>(layer.output_size) ||
                    layer.biases.size() != static_cast<std::size_t>(layer.output_size))
                    throw std::runtime_error("invalid dense layer dimensions");
                for (const std::vector<double> &row : layer.weights)
                    if (row.size() != static_cast<std::size_t>(layer.input_size))
                        throw std::runtime_error("invalid dense layer weight dimensions");
            }
        }

        int first_linear_or_shape(bool input) const
        {
            if (!input)
            {
                for (std::size_t index = operations_.size(); index > 0; --index)
                {
                    const Operation &operation = operations_[index - 1];
                    if (operation.kind == OperationKind::Linear)
                        return operation.linear_layer.output_size;
                    if (!operation.output_shape.empty()) return shape_size(operation.output_shape);
                }
                return 0;
            }
            for (const Operation &operation : operations_)
            {
                if (operation.kind == OperationKind::Linear)
                    return operation.linear_layer.input_size;
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
