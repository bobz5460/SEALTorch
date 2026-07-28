#pragma once

#include <SEALTorch/model.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace sealtorch
{
    // Native ciphertexts are provider-specific. This API deliberately owns
    // them, so callers can switch providers without changing their inference
    // flow or accidentally mixing incompatible ciphertext formats.
    enum class InferenceProvider { Auto, CPU, CUDA };
    enum class CiphertextLayout { Packed, Scalar };

    struct EncryptedInferenceOptions
    {
        InferenceProvider provider = InferenceProvider::Auto;
        CiphertextLayout layout = CiphertextLayout::Packed;
        std::size_t thread_count = 4;
        std::size_t ring_dimension = 16384;
        std::size_t multiplicative_depth = 15;
        std::size_t scaling_modulus_bits = 40;
        std::size_t first_modulus_bits = 50;
        std::size_t scale_bits = 25;
        int cuda_device = 0;
    };

    struct EncryptedInferenceResult
    {
        std::vector<double> values;
        double encrypt_ms = 0.0;
        double evaluate_ms = 0.0;
        double decrypt_ms = 0.0;
        std::size_t input_ciphertext_bytes = 0;
        std::size_t output_ciphertext_bytes = 0;
    };

    // One encrypt -> evaluate -> decrypt API for Microsoft SEAL (CPU) and
    // FIDESlib (CUDA). Add model operations to native backends while keeping
    // application code independent of the selected HE provider.
    class EncryptedInference
    {
    public:
        EncryptedInference(Sequential model, EncryptedInferenceOptions options = {});
        ~EncryptedInference();
        EncryptedInference(EncryptedInference &&) noexcept;
        EncryptedInference &operator=(EncryptedInference &&) noexcept;
        EncryptedInference(const EncryptedInference &) = delete;
        EncryptedInference &operator=(const EncryptedInference &) = delete;

        EncryptedInferenceResult predict(const std::vector<double> &input);
        InferenceProvider provider() const;
        static bool cuda_available();

    private:
        struct Implementation;
        std::unique_ptr<Implementation> implementation_;
    };
}
