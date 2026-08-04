#include <SEALTorch/sealtorch.h>

#include <cassert>
#include <cmath>

int main()
{
    seal::EncryptionParameters parameters(seal::scheme_type::ckks);
    parameters.set_poly_modulus_degree(8192);
    parameters.set_coeff_modulus(
        seal::CoeffModulus::Create(8192, {50, 40, 40}));
    seal::SEALContext context(parameters, true, seal::sec_level_type::none);
    seal::KeyGenerator keys(context);
    seal::RelinKeys relin_keys;
    seal::GaloisKeys galois_keys;
    keys.create_relin_keys(relin_keys);
    keys.create_galois_keys(std::vector<int>{-2, 1}, galois_keys);

    seal::CKKSEncoder encoder(context);
    seal::Evaluator evaluator(context);
    const double scale = std::ldexp(1.0, 40);

    seal::Plaintext plain;
    encoder.encode(std::vector<double>{1.0, 2.0}, scale, plain);
    seal::Ciphertext encrypted;
    seal::Encryptor(context, keys.secret_key()).encrypt_symmetric(
        plain, encrypted);

    sealtorch::DenseLayer layer{
        2, 2, {{1.0, 2.0}, {3.0, 4.0}}, {0.0, 0.0}};
    sealtorch::Sequential sequential;
    sequential.add(std::move(layer));
    sealtorch::SealCiphertextModel model(std::move(sequential));
    const sealtorch::SealInferenceConfig config{
        context, evaluator, relin_keys, galois_keys, encoder, scale, 2};

    const seal::Ciphertext output = model.predict(encrypted, config);

    seal::Plaintext decoded;
    seal::Decryptor(context, keys.secret_key()).decrypt(output, decoded);
    std::vector<double> values;
    encoder.decode(decoded, values);
    assert(std::abs(values[0] - 5.0) < 0.01);
    assert(std::abs(values[1] - 11.0) < 0.01);
}
