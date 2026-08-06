#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sealtorch {

struct DenseLayer {
    int input_size = 0;
    int output_size = 0;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

enum class OperationKind { Linear, Activation };

struct Operation {
    OperationKind kind;
    DenseLayer linear;
};

class Model {
  public:
    Model& add(DenseLayer layer) {
        if (layer.input_size <= 0 || layer.output_size <= 0 ||
            layer.weights.size() != static_cast<std::size_t>(layer.output_size) ||
            layer.biases.size() != static_cast<std::size_t>(layer.output_size))
            throw std::runtime_error("invalid linear layer");
        for (const auto& row : layer.weights)
            if (row.size() != static_cast<std::size_t>(layer.input_size))
                throw std::runtime_error("invalid linear layer weights");
        if (!operations_.empty()) {
            const DenseLayer* previous = last_linear();
            if (previous && previous->output_size != layer.input_size)
                throw std::runtime_error("linear layer dimensions do not connect");
        }
        operations_.push_back({ OperationKind::Linear, std::move(layer) });
        return *this;
    }

    Model& activation() {
        if (operations_.empty() || operations_.back().kind != OperationKind::Linear)
            throw std::runtime_error("activation must follow a linear layer");
        operations_.push_back({ OperationKind::Activation, {} });
        return *this;
    }

    const std::vector<Operation>& operations() const { return operations_; }
    int input_size() const { return operations_.empty() ? 0 : operations_.front().linear.input_size; }
    int output_size() const { const auto* layer = last_linear(); return layer ? layer->output_size : 0; }
    std::size_t activation_count() const {
        std::size_t count = 0;
        for (const auto& operation : operations_)
            count += operation.kind == OperationKind::Activation;
        return count;
    }

  private:
    const DenseLayer* last_linear() const {
        for (auto operation = operations_.rbegin(); operation != operations_.rend(); ++operation)
            if (operation->kind == OperationKind::Linear) return &operation->linear;
        return nullptr;
    }
    std::vector<Operation> operations_;
};

} // namespace sealtorch
