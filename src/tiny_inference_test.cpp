#include <SEALTorch/evaluator.h>

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    // Deliberately tiny two-layer network. The second layer is important:
    // its weights are multiplied after the first layer has been rescaled,
    // which is the case that used to trigger the NTT parameter mismatch.
    NeuralNetwork model;
    model.input_size = 2;
    model.output_size = 1;
    model.layers = {
        DenseLayer{2, 2, {{1.0, 0.5}, {-0.5, 1.0}}, {0.0, 0.0}},
        DenseLayer{2, 1, {{0.75, -0.25}}, {0.1}}};

    seal::EncryptionParameters parameters(seal::scheme_type::ckks);
    parameters.set_poly_modulus_degree(16384);
    parameters.set_coeff_modulus(
        seal::CoeffModulus::Create(16384, {60, 40, 40, 40, 40, 40, 40, 60}));
    seal::SEALContext context(parameters);
    seal::KeyGenerator key_generator(context);
    seal::Encryptor encryptor(context, key_generator.secret_key());
    seal::Decryptor decryptor(context, key_generator.secret_key());
    seal::Evaluator evaluator(context);
    seal::CKKSEncoder encoder(context);
    seal::RelinKeys relin_keys;
    key_generator.create_relin_keys(relin_keys);

    constexpr double scale = static_cast<double>(1ULL << 30);
    const std::vector<double> input = {0.25, -0.5};
    std::vector<seal::Ciphertext> encrypted_input(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        seal::Plaintext plain;
        encoder.encode(input[index], scale, plain);
        encryptor.encrypt_symmetric(plain, encrypted_input[index]);
    }

    sealtorch::Evaluator model_evaluator(model);
    const auto encrypted_output = model_evaluator.predict(
        encrypted_input, evaluator, relin_keys, encoder, scale);

    seal::Plaintext plain_output;
    decryptor.decrypt(encrypted_output.front(), plain_output);
    std::vector<double> decoded;
    encoder.decode(plain_output, decoded);
    if (decoded.empty() || !std::isfinite(decoded.front()))
    {
        std::cerr << "tiny encrypted inference returned no finite output\n";
        return 1;
    }

    std::cout << "tiny encrypted inference output: " << decoded.front() << '\n';
    return 0;
}
