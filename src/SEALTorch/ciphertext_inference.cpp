#include <SEALTorch/ciphertext_inference.h>

#include <SEALTorch/cuda_ciphertext_inference.h>
#include <SEALTorch/packing.h>
#include <SEALTorch/seal_ciphertext_backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace sealtorch
{
    namespace
    {
        template <class Serializable>
        std::size_t serialized_bytes(const Serializable &value)
        {
            std::ostringstream stream(std::ios::out | std::ios::binary);
            value.save(stream);
            return stream.tellp() < 0 ? 0 : static_cast<std::size_t>(stream.tellp());
        }

        std::size_t ciphertext_memory_bytes(const seal::Ciphertext &value, const seal::SEALContext &context)
        {
            const auto context_data = context.get_context_data(value.parms_id());
            if (!context_data) return 0;
            return value.size() * context_data->parms().poly_modulus_degree() *
                context_data->parms().coeff_modulus().size() * sizeof(std::uint64_t);
        }

        seal::SEALContext make_cpu_context(const CiphertextInferenceOptions &options)
        {
            if (options.ring_dimension < 1024 || (options.ring_dimension & (options.ring_dimension - 1)) ||
                options.multiplicative_depth == 0 || options.scale_bits == 0)
                throw std::runtime_error("invalid ciphertext inference options");
            seal::EncryptionParameters parameters(seal::scheme_type::ckks);
            parameters.set_poly_modulus_degree(options.ring_dimension);
            std::vector<int> modulus_bits(options.multiplicative_depth, static_cast<int>(options.scale_bits));
            modulus_bits.insert(modulus_bits.begin(), static_cast<int>(options.first_modulus_bits));
            parameters.set_coeff_modulus(seal::CoeffModulus::Create(options.ring_dimension, modulus_bits));
            seal::SEALContext context(parameters, true, seal::sec_level_type::none);
            if (!context.parameters_set())
                throw std::runtime_error("invalid CKKS parameters for the Microsoft SEAL CPU backend");
            return context;
        }

        class CpuCiphertextInference
        {
        public:
            CpuCiphertextInference(Sequential model, CiphertextInferenceOptions options)
                : options_(std::move(options)), context_(make_cpu_context(options_)), keys_(context_),
                  encryptor_(context_, keys_.secret_key()), evaluator_(context_), encoder_(context_),
                  model_(std::move(model)), scale_(std::ldexp(1.0, static_cast<int>(options_.scale_bits)))
            {
                keys_.create_relin_keys(relin_keys_);
                std::vector<std::int32_t> rotations;
                const std::size_t slot_count = encoder_.slot_count();
                for (const Operation &operation : model_.model().operations())
                {
                    if (operation.kind != OperationKind::Linear)
                        continue;

                    for (std::size_t diagonal :
                         active_diagonals(operation.linear_layer, slot_count))
                    {
                        const DiagonalSplit split =
                            split_diagonal(diagonal, slot_count);
                        const std::int32_t baby_rotation =
                            signed_rotation(split.baby, slot_count);
                        const std::int32_t giant_rotation =
                            signed_rotation(split.giant, slot_count);
                        if (baby_rotation != 0)
                            rotations.push_back(baby_rotation);
                        if (giant_rotation != 0)
                            rotations.push_back(giant_rotation);
                    }
                }
                std::sort(rotations.begin(), rotations.end());
                rotations.erase(std::unique(rotations.begin(), rotations.end()), rotations.end());
                keys_.create_galois_keys(rotations, galois_keys_);
                secret_key_bytes_ = serialized_bytes(keys_.secret_key());
                relin_keys_bytes_ = serialized_bytes(relin_keys_);
                galois_keys_bytes_ = serialized_bytes(galois_keys_);
            }

            CiphertextInferenceResult predict(const std::vector<double> &input)
            {
                CiphertextInferenceResult result;
                const auto encrypt_start = std::chrono::steady_clock::now();
                seal::Plaintext plain;
                encoder_.encode(input, scale_, plain);
                seal::Ciphertext encrypted;
                encryptor_.encrypt_symmetric(plain, encrypted);
                result.input_ciphertext_bytes = serialized_bytes(encrypted);
                result.input_ciphertext_memory_bytes = ciphertext_memory_bytes(encrypted, context_);
                result.secret_key_bytes = secret_key_bytes_;
                result.relin_keys_bytes = relin_keys_bytes_;
                result.galois_keys_bytes = galois_keys_bytes_;
                const auto encrypt_end = std::chrono::steady_clock::now();

                const SealCiphertextBackend &backend = options_.layout == CiphertextLayout::Packed
                    ? static_cast<const SealCiphertextBackend &>(packed_backend_)
                    : static_cast<const SealCiphertextBackend &>(scalar_backend_);
                SealInferenceConfig config(
                    context_, evaluator_, relin_keys_, galois_keys_, encoder_, scale_, backend,
                    options_.thread_count);
                const auto evaluate_start = std::chrono::steady_clock::now();
                const auto output = model_.predict({encrypted}, config);
                const auto evaluate_end = std::chrono::steady_clock::now();

                const auto decrypt_start = std::chrono::steady_clock::now();
                seal::Decryptor decryptor(context_, keys_.secret_key());
                if (options_.layout == CiphertextLayout::Packed)
                {
                    seal::Plaintext decoded;
                    decryptor.decrypt(output.front(), decoded);
                    encoder_.decode(decoded, result.values);
                    result.output_ciphertext_bytes = serialized_bytes(output.front());
                    result.output_ciphertext_memory_bytes = ciphertext_memory_bytes(output.front(), context_);
                }
                else for (const seal::Ciphertext &source : output)
                {
                    seal::Plaintext decoded;
                    std::vector<double> values;
                    decryptor.decrypt(source, decoded);
                    encoder_.decode(decoded, values);
                    result.values.push_back(values.front());
                    result.output_ciphertext_bytes += serialized_bytes(source);
                    result.output_ciphertext_memory_bytes += ciphertext_memory_bytes(source, context_);
                }
                result.values.resize(static_cast<std::size_t>(model_.model().output_size()));
                const auto decrypt_end = std::chrono::steady_clock::now();
                result.encrypt_ms = std::chrono::duration<double, std::milli>(encrypt_end - encrypt_start).count();
                result.evaluate_ms = std::chrono::duration<double, std::milli>(evaluate_end - evaluate_start).count();
                result.decrypt_ms = std::chrono::duration<double, std::milli>(decrypt_end - decrypt_start).count();
                return result;
            }

        private:
            CiphertextInferenceOptions options_;
            seal::SEALContext context_;
            seal::KeyGenerator keys_;
            seal::RelinKeys relin_keys_;
            seal::GaloisKeys galois_keys_;
            seal::Encryptor encryptor_;
            seal::Evaluator evaluator_;
            seal::CKKSEncoder encoder_;
            SealScalarBackend scalar_backend_;
            SealPackedBackend packed_backend_;
            SealCiphertextModel model_;
            double scale_;
            std::size_t secret_key_bytes_ = 0;
            std::size_t relin_keys_bytes_ = 0;
            std::size_t galois_keys_bytes_ = 0;
        };
    }

    struct CiphertextInference::Implementation
    {
        ExecutionTarget target;
        std::unique_ptr<CpuCiphertextInference> cpu_inference;
        std::unique_ptr<cuda::CiphertextInferenceEngine> cuda_engine;
        CiphertextInferenceOptions options;

        Implementation(Sequential model, CiphertextInferenceOptions options_value)
            : options(std::move(options_value))
        {
            if (options.thread_count == 0)
                throw std::runtime_error("thread count must be greater than zero");
            target = options.target == ExecutionTarget::Auto
                ? (cuda::cuda_available() ? ExecutionTarget::Cuda : ExecutionTarget::Cpu)
                : options.target;
            if (target == ExecutionTarget::Cuda)
            {
                if (options.layout != CiphertextLayout::Packed)
                    throw std::runtime_error("CUDA encrypted inference currently supports packed ciphertexts only");
                cuda::CiphertextInferenceOptions cuda_options{
                    options.thread_count,
                    options.ring_dimension,
                    options.multiplicative_depth,
                    options.scaling_modulus_bits,
                    options.first_modulus_bits,
                    options.cuda_device,
                };
                cuda_engine =
                    std::make_unique<cuda::CiphertextInferenceEngine>(
                        std::move(model), cuda_options);
            }
            else
                cpu_inference = std::make_unique<CpuCiphertextInference>(std::move(model), options);
        }

        CiphertextInferenceResult predict(const std::vector<double> &input)
        {
            if (cpu_inference) return cpu_inference->predict(input);
            CiphertextInferenceResult result;
            const cuda::CiphertextInferenceResult cuda_result = cuda_engine->predict(input);
            result.values = cuda_result.values;
            result.encrypt_ms = cuda_result.encrypt_ms;
            result.evaluate_ms = cuda_result.evaluate_ms;
            result.decrypt_ms = cuda_result.decrypt_ms;
            // FIDESlib keeps ciphertexts opaque, so exposing a SEAL-derived
            // estimate here would be inaccurate. A value of zero means the
            // provider does not currently expose this measurement.
            return result;
        }
    };

    CiphertextInference::CiphertextInference(Sequential model, CiphertextInferenceOptions options)
        : implementation_(std::make_unique<Implementation>(std::move(model), std::move(options))) {}
    CiphertextInference::~CiphertextInference() = default;
    CiphertextInference::CiphertextInference(CiphertextInference &&) noexcept = default;
    CiphertextInference &CiphertextInference::operator=(CiphertextInference &&) noexcept = default;
    CiphertextInferenceResult CiphertextInference::predict(
        const std::vector<double> &input)
    {
        return implementation_->predict(input);
    }
    ExecutionTarget CiphertextInference::target() const { return implementation_->target; }
    bool CiphertextInference::cuda_available() { return cuda::cuda_available(); }
}
