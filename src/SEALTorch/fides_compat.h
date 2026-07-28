#pragma once

// Small SEAL-shaped facade over FIDESlib.  It keeps the model evaluator
// independent of a particular FHE provider while routing every operation to
// FIDESlib/OpenFHE (and therefore its CUDA backend).
#include <fideslib.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace seal {
using parms_id_type = std::size_t;

class SEALContext {
public:
    explicit SEALContext(fideslib::CryptoContext<fideslib::DCRTPoly> value) : value_(std::move(value)) {}
    const fideslib::CryptoContext<fideslib::DCRTPoly> &value() const { return value_; }
private:
    fideslib::CryptoContext<fideslib::DCRTPoly> value_;
};

class Plaintext {
public:
    fideslib::Plaintext value;
    parms_id_type parms_id() const { return value ? value->GetLevel() : 0; }
};

class Ciphertext {
public:
    fideslib::Ciphertext<fideslib::DCRTPoly> value;
    double &scale() { return scale_; }
    double scale() const { return scale_; }
    parms_id_type parms_id() const { return value ? value->GetLevel() : 0; }
private:
    double scale_ = 1.0;
};

class RelinKeys {};
class GaloisKeys {};
class MemoryPoolHandle { public: static MemoryPoolHandle ThreadLocal() { return {}; } };

class CKKSEncoder {
public:
    explicit CKKSEncoder(const SEALContext &context) : context_(context.value()) {}
    std::size_t slot_count() const { return context_->GetRingDimension() / 2; }
    void encode(const std::vector<double> &values, double scale, Plaintext &output, MemoryPoolHandle = {}) const {
        output.value = context_->MakeCKKSPackedPlaintext(values);
        (void)scale;
    }
    void encode(double value, double scale, Plaintext &output, MemoryPoolHandle = {}) const {
        encode(std::vector<double>{value}, scale, output);
    }
    void decode(const Plaintext &input, std::vector<double> &output) const { output = input.value->GetRealPackedValue(); }
private:
    fideslib::CryptoContext<fideslib::DCRTPoly> context_;
};

class Evaluator {
public:
    explicit Evaluator(const SEALContext &context) : context_(context.value()) {}
    void multiply_plain(const Ciphertext &input, const Plaintext &plain, Ciphertext &output, MemoryPoolHandle = {}) const {
        auto copy = plain.value; output.value = context_->EvalMult(input.value, copy);
    }
    void add_inplace(Ciphertext &left, const Ciphertext &right) const { context_->EvalAddInPlace(left.value, right.value); }
    void add_plain_inplace(Ciphertext &left, const Plaintext &right) const { auto copy = right.value; context_->EvalAddInPlace(left.value, copy); }
    void rotate_vector(const Ciphertext &input, int step, const GaloisKeys &, Ciphertext &output, MemoryPoolHandle = {}) const { output.value = context_->EvalRotate(input.value, step); }
    void square(const Ciphertext &input, Ciphertext &output) const { output.value = context_->EvalSquare(input.value); }
    void relinearize_inplace(Ciphertext &, const RelinKeys &) const {}
    void rescale_to_next_inplace(Ciphertext &value) const { context_->RescaleInPlace(value.value); }
    void mod_switch_to_inplace(Plaintext &, parms_id_type) const {}
    void mod_switch_to_inplace(Ciphertext &, parms_id_type) const {}
private:
    fideslib::CryptoContext<fideslib::DCRTPoly> context_;
};

class KeyGenerator {
public:
    explicit KeyGenerator(const SEALContext &context) : context_(context.value()), keys_(context_->KeyGen()) {}
    const fideslib::PrivateKey<fideslib::DCRTPoly> &secret_key() const { return keys_.secretKey; }
    const fideslib::PublicKey<fideslib::DCRTPoly> &public_key() const { return keys_.publicKey; }
    void create_relin_keys(RelinKeys &) { context_->EvalMultKeyGen(keys_.secretKey); }
    void create_galois_keys(GaloisKeys &, const std::vector<int32_t> &steps) { context_->EvalRotateKeyGen(keys_.secretKey, steps); }
    void load_context() { context_->LoadContext(keys_.publicKey); }
private:
    fideslib::CryptoContext<fideslib::DCRTPoly> context_;
    fideslib::KeyPair<fideslib::DCRTPoly> keys_;
};

class Encryptor {
public:
    Encryptor(const SEALContext &context, const fideslib::PrivateKey<fideslib::DCRTPoly> &key) : context_(context.value()), key_(key) {}
    void encrypt_symmetric(Plaintext &input, Ciphertext &output) const { output.value = context_->Encrypt(key_, input.value); }
private:
    fideslib::CryptoContext<fideslib::DCRTPoly> context_;
    fideslib::PrivateKey<fideslib::DCRTPoly> key_;
};

class Decryptor {
public:
    Decryptor(const SEALContext &context, const fideslib::PrivateKey<fideslib::DCRTPoly> &key) : context_(context.value()), key_(key) {}
    void decrypt(Ciphertext &input, Plaintext &output) const { context_->Decrypt(key_, input.value, &output.value); }
private:
    fideslib::CryptoContext<fideslib::DCRTPoly> context_;
    fideslib::PrivateKey<fideslib::DCRTPoly> key_;
};
} // namespace seal
