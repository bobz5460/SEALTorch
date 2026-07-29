#include <SEALTorch/ciphertext_inference.h>
#include <SEALTorch/model.h>

#include <cassert>
#include <cmath>
#include <vector>

int main()
{
    sealtorch::DenseLayer layer;
    layer.input_size = 2;
    layer.output_size = 2;
    layer.weights = {
        {1.0, 2.0},
        {3.0, 4.0},
    };
    layer.biases = {0.0, 0.0};

    sealtorch::Sequential model;
    model.add(sealtorch::Linear(layer));

    sealtorch::CiphertextInferenceOptions options;
    options.target = sealtorch::ExecutionTarget::Cpu;
    options.layout = sealtorch::CiphertextLayout::Packed;
    options.thread_count = 2;
    options.ring_dimension = 8192;
    options.multiplicative_depth = 2;
    options.first_modulus_bits = 50;
    options.scale_bits = 40;

    sealtorch::CiphertextInference inference(model, options);
    const sealtorch::CiphertextInferenceResult result =
        inference.predict({1.0, 2.0});

    assert(result.values.size() == 2);
    assert(std::abs(result.values[0] - 5.0) < 0.01);
    assert(std::abs(result.values[1] - 11.0) < 0.01);
}
