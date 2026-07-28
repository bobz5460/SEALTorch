#pragma once

#include <SEALTorch/model.h>

#include <fideslib.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sealtorch::cuda
{
    // True when FIDESlib can use at least one CUDA device in this process.
    bool cuda_available();

    // Parameters needed to create a native FIDESlib CKKS context. Keep this
    // separate from a model so experiments can vary HE parameters clearly.
    struct CiphertextInferenceOptions
    {
        std::size_t ring_dimension = 16384;
        std::size_t multiplicative_depth = 15;
        std::size_t scaling_modulus_bits = 40;
        std::size_t first_modulus_bits = 50;
        int device = 0;
    };

    // Packed ciphertext inference backed directly by FIDESlib. The engine
    // owns its context, keys, and cached model plaintexts; callers only pass
    // ordinary doubles in and receive ordinary doubles out.
    class CiphertextInferenceEngine
    {
    public:
        CiphertextInferenceEngine(Sequential model, CiphertextInferenceOptions options = {});
        ~CiphertextInferenceEngine();
        CiphertextInferenceEngine(CiphertextInferenceEngine &&) noexcept;
        CiphertextInferenceEngine &operator=(CiphertextInferenceEngine &&) noexcept;
        CiphertextInferenceEngine(const CiphertextInferenceEngine &) = delete;
        CiphertextInferenceEngine &operator=(const CiphertextInferenceEngine &) = delete;

        std::vector<double> predict(const std::vector<double> &input);
        std::size_t slot_count() const;

    private:
        struct Implementation;
        std::unique_ptr<Implementation> implementation_;
    };
}
