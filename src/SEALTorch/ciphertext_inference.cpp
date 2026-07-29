#include <SEALTorch/ciphertext_inference.h>

#include <SEALTorch/cuda_ciphertext_inference.h>
#include <SEALTorch/seal_ciphertext_backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sealtorch
{
    namespace
    {
        std::size_t ciphertext_bytes(const CiphertextInferenceOptions &options)
        {
            // A provider-neutral coefficient-storage estimate. Native
            // serialized ciphertexts differ, so reporting one common metric
            // makes CPU/CUDA experiment results comparable.
            return 2 * options.ring_dimension * (options.multiplicative_depth + 1) * sizeof(std::uint64_t);
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
                std::vector<int32_t> rotations;
                for (const DenseLayer &layer : model_.model().layers())
                {
                    for (int step = 1; step < layer.input_size; ++step) rotations.push_back(step);
                    for (int step = 1; step < layer.output_size; ++step) rotations.push_back(-step);
                }
                std::sort(rotations.begin(), rotations.end());
                rotations.erase(std::unique(rotations.begin(), rotations.end()), rotations.end());
                keys_.create_galois_keys(rotations, galois_keys_);
            }

            CiphertextInferenceResult predict(const std::vector<double> &input)
            {
                CiphertextInferenceResult result;
                const auto encrypt_start = std::chrono::steady_clock::now();
                seal::Plaintext plain;
                encoder_.encode(input, scale_, plain);
                seal::Ciphertext encrypted;
                encryptor_.encrypt_symmetric(plain, encrypted);
                result.input_ciphertext_bytes = ciphertext_bytes(options_);
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
                    result.output_ciphertext_bytes = ciphertext_bytes(options_);
                }
                else for (const seal::Ciphertext &source : output)
                {
                    seal::Plaintext decoded;
                    std::vector<double> values;
                    decryptor.decrypt(source, decoded);
                    encoder_.decode(decoded, values);
                    result.values.push_back(values.front());
                    result.output_ciphertext_bytes += ciphertext_bytes(options_);
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
                cuda_engine = std::make_unique<cuda::CiphertextInferenceEngine>(std::move(model), cuda::CiphertextInferenceOptions{
                    options.ring_dimension, options.multiplicative_depth, options.scaling_modulus_bits,
                    options.first_modulus_bits, options.cuda_device});
            }
            else
                cpu_inference = std::make_unique<CpuCiphertextInference>(std::move(model), options);
        }

        CiphertextInferenceResult predict(const std::vector<double> &input)
        {
            if (cpu_inference) return cpu_inference->predict(input);
            const auto started = std::chrono::steady_clock::now();
            CiphertextInferenceResult result;
            result.values = cuda_engine->predict(input);
            result.evaluate_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            result.input_ciphertext_bytes = ciphertext_bytes(options);
            result.output_ciphertext_bytes = ciphertext_bytes(options);
            return result;
        }
    };

    CiphertextInference::CiphertextInference(Sequential model, CiphertextInferenceOptions options)
        : implementation_(std::make_unique<Implementation>(std::move(model), std::move(options))) {}
    CiphertextInference::~CiphertextInference() = default;
    CiphertextInference::CiphertextInference(CiphertextInference &&) noexcept = default;
    CiphertextInference &CiphertextInference::operator=(CiphertextInference &&) noexcept = default;
    CiphertextInferenceResult CiphertextInference::predict(const std::vector<double> &input) { return implementation_->predict(input); }
    ExecutionTarget CiphertextInference::target() const { return implementation_->target; }
    bool CiphertextInference::cuda_available() { return cuda::cuda_available(); }
}
