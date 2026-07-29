#include <SEALTorch/cuda_ciphertext_inference.h>

#include <stdexcept>
#include <utility>

namespace sealtorch::cuda
{
    bool cuda_available()
    {
        return false;
    }

    struct CiphertextInferenceEngine::Implementation
    {
    };

    CiphertextInferenceEngine::CiphertextInferenceEngine(
        Sequential,
        CiphertextInferenceOptions)
    {
        throw std::runtime_error(
            "CUDA inference is unavailable because this build does not "
            "include FIDESlib");
    }

    CiphertextInferenceEngine::~CiphertextInferenceEngine() = default;
    CiphertextInferenceEngine::CiphertextInferenceEngine(
        CiphertextInferenceEngine &&) noexcept = default;
    CiphertextInferenceEngine &CiphertextInferenceEngine::operator=(
        CiphertextInferenceEngine &&) noexcept = default;

    CiphertextInferenceResult CiphertextInferenceEngine::predict(
        const std::vector<double> &)
    {
        throw std::runtime_error("CUDA inference is unavailable");
    }

    std::size_t CiphertextInferenceEngine::slot_count() const
    {
        return 0;
    }
}
