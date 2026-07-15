 #pragma once

 #include <cstddef>
 #include <functional>
 #include <stdexcept>
 #include <utility>
 #include <vector>

struct DenseLayer
{
    int input_size;
    int output_size;

    // weights[output_neuron][input_neuron]
    std::vector<std::vector<double>> weights;

    std::vector<double> biases;

};

struct NeuralNetwork
{
    int input_size;
    int output_size;

    std::vector<DenseLayer> layers;
};