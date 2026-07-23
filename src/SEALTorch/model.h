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

    enum class ActivationType
    {
        Relu,
        Gelu
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

        Linear(DenseLayer layer) : layer_(std::move(layer)) {}

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

        Sequential &add(const Linear &layer)
        {
            if (layers_.empty()) input_size_ = layer.parameters().input_size;
            else if (layers_.back().output_size != layer.parameters().input_size)
                throw std::runtime_error("linear layer sizes do not match");
            layers_.push_back(layer.parameters());
            activations_.push_back(false);
            activation_types_.push_back(ActivationType::Relu);
            output_size_ = layer.parameters().output_size;
            return *this;
        }

        Sequential &add(const Activation &activation)
        {
            if (layers_.empty()) throw std::runtime_error("activation needs a linear layer");
            activations_.back() = true;
            activation_types_.back() = activation.type();
            return *this;
        }

        const std::vector<DenseLayer> &layers() const { return layers_; }
        int input_size() const { return input_size_; }
        int output_size() const { return output_size_; }
        bool has_activation(std::size_t index) const { return activations_[index]; }
        ActivationType activation(std::size_t index) const { return activation_types_[index]; }

    private:
        int input_size_ = 0;
        int output_size_ = 0;
        std::vector<DenseLayer> layers_;
        std::vector<bool> activations_;
        std::vector<ActivationType> activation_types_;
    };

    using NeuralNetwork = Sequential;
}
