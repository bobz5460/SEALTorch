#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sealtorch
{
    struct DenseLayer
    {
        int input_size = 0;
        int output_size = 0;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
    };

    enum class ActivationType { Relu, Gelu, Tanh };
    enum class OperationKind { Linear, Activation };

    struct Operation
    {
        OperationKind kind = OperationKind::Linear;
        DenseLayer linear_layer;
        ActivationType activation_type = ActivationType::Relu;
    };

    class Sequential
    {
    public:
        Sequential &add(DenseLayer layer)
        {
            if (layer.input_size <= 0 || layer.output_size <= 0 ||
                layer.weights.size() != static_cast<std::size_t>(layer.output_size) ||
                layer.biases.size() != static_cast<std::size_t>(layer.output_size))
                throw std::runtime_error("invalid linear layer");
            for (const auto &row : layer.weights)
                if (row.size() != static_cast<std::size_t>(layer.input_size))
                    throw std::runtime_error("invalid linear layer");
            if (!operations_.empty()) {
                const DenseLayer *previous = last_linear();
                if (previous && previous->output_size != layer.input_size)
                    throw std::runtime_error("linear layer sizes do not match");
            }
            operations_.push_back({OperationKind::Linear, std::move(layer), {}});
            return *this;
        }

        Sequential &add(ActivationType activation)
        {
            if (operations_.empty() ||
                operations_.back().kind != OperationKind::Linear)
                throw std::runtime_error("activation needs a preceding linear layer");
            operations_.push_back({OperationKind::Activation, {}, activation});
            return *this;
        }

        const std::vector<Operation> &operations() const { return operations_; }

        int input_size() const
        {
            for (const auto &operation : operations_)
                if (operation.kind == OperationKind::Linear)
                    return operation.linear_layer.input_size;
            return 0;
        }

        int output_size() const
        {
            const DenseLayer *layer = last_linear();
            return layer ? layer->output_size : 0;
        }

    private:
        const DenseLayer *last_linear() const
        {
            for (auto operation = operations_.rbegin();
                 operation != operations_.rend(); ++operation)
                if (operation->kind == OperationKind::Linear)
                    return &operation->linear_layer;
            return nullptr;
        }

        std::vector<Operation> operations_;
    };
}
