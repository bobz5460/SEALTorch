#pragma once

#include "model.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace sealtorch {

struct InferenceOptions {
    std::size_t ring_dimension = 32768;
    std::size_t multiplicative_depth = 12;
    std::size_t scaling_modulus_bits = 40;
    std::size_t first_modulus_bits = 50;
    double lower_bound = -1.0;
    double upper_bound = 1.0;
    std::vector<double> chebyshev_coefficients;
    int device = 0;
    bool stream_weights = true;
};

struct InferenceResult {
    std::vector<double> values;
    double encrypt_ms = 0.0;
    double evaluate_ms = 0.0;
    double decrypt_ms = 0.0;
    std::size_t idle_ram_bytes = 0;
    std::size_t ram_bytes = 0;
    std::size_t idle_vram_bytes = 0;
    std::size_t vram_bytes = 0;
};

class InferenceEngine {
  public:
    InferenceEngine(Model model, InferenceOptions options);
    ~InferenceEngine();
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    InferenceResult predict(const std::vector<double>& input);
    std::size_t slot_count() const;
    static bool available();

  private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace sealtorch
