#include <SEALTorch/evaluator.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <cctype>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {
struct Value { enum class Kind { null_value, number, string, array, object } kind = Kind::null_value; double number = 0; std::string string; std::vector<Value> array; std::map<std::string, Value> object; const Value &at(const std::string &key) const { return object.at(key); } };
class Parser {
public: explicit Parser(std::string text) : text_(std::move(text)) {} Value parse() { skip(); return value(); }
private: std::string text_; std::size_t position_ = 0;
    void skip() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
    void expect(char c) { skip(); if (position_ >= text_.size() || text_[position_++] != c) throw std::runtime_error("invalid JSON"); }
    std::string string_value() { expect('"'); std::string result; while (position_ < text_.size() && text_[position_] != '"') { if (text_[position_] == '\\' && position_ + 1 < text_.size()) ++position_; result += text_[position_++]; } expect('"'); return result; }
    Value value() { skip(); if (position_ >= text_.size()) throw std::runtime_error("unexpected end of JSON"); if (text_[position_] == '{') return object_value(); if (text_[position_] == '[') return array_value(); if (text_[position_] == '"') { Value result; result.kind=Value::Kind::string; result.string=string_value(); return result; } auto start=position_; while (position_<text_.size() && std::string(",]} \t\r\n").find(text_[position_])==std::string::npos) ++position_; auto token=text_.substr(start,position_-start); if(token=="null"||token=="true"||token=="false") return {}; Value result; result.kind=Value::Kind::number; result.number=std::stod(token); return result; }
    Value array_value() { expect('['); Value result; result.kind=Value::Kind::array; skip(); if(position_<text_.size()&&text_[position_]==']'){++position_;return result;} for(;;){result.array.push_back(value());skip();if(position_<text_.size()&&text_[position_]==']'){++position_;return result;}expect(',');} }
    Value object_value() { expect('{'); Value result; result.kind=Value::Kind::object; skip(); if(position_<text_.size()&&text_[position_]=='}'){++position_;return result;} for(;;){auto key=string_value();expect(':');result.object.emplace(key,value());skip();if(position_<text_.size()&&text_[position_]=='}'){++position_;return result;}expect(',');} }
};
}

static NeuralNetwork load_model(const std::string &path) {
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf(); auto root=json::Parser(contents.str()).parse(); const auto &tensors=root.at("tensors").object;
    NeuralNetwork model{784,10,{}};
    for (const auto &names : {std::pair<std::string,std::string>{"network.1.weight","network.1.bias"},{"network.3.weight","network.3.bias"},{"network.5.weight","network.5.bias"}}) {
        const auto &weights=tensors.at(names.first).at("data").array; const auto &biases=tensors.at(names.second).at("data").array; DenseLayer layer{(int)weights.front().array.size(),(int)weights.size(),{}, {}}; layer.weights.resize(weights.size()); layer.biases.resize(biases.size());
        for(size_t o=0;o<weights.size();++o){for(const auto &w:weights[o].array)layer.weights[o].push_back(w.number);layer.biases[o]=biases[o].number;} model.layers.push_back(std::move(layer));
    } return model;
}
static double gelu(double x) { return 0.06762090 + .5*x + .48257484*x*x - .06659632*x*x*x*x; }
static std::vector<double> plain_predict(const NeuralNetwork &model, const std::vector<double> &input, bool relu) {
    auto values=input; for(size_t l=0;l<model.layers.size();++l){const auto &layer=model.layers[l];std::vector<double> next(layer.output_size);for(int o=0;o<layer.output_size;++o){next[o]=layer.biases[o];for(int i=0;i<layer.input_size;++i)next[o]+=layer.weights[o][i]*values[i];if(l+1<model.layers.size())next[o]=relu?std::max(0.0,next[o]):gelu(next[o]);}values=std::move(next);}return values;
}
static uint32_t read_u32(std::ifstream &file){unsigned char b[4];if(!file.read((char*)b,4))throw std::runtime_error("truncated MNIST file");return (uint32_t(b[0])<<24)|(uint32_t(b[1])<<16)|(uint32_t(b[2])<<8)|b[3];}
static std::vector<std::vector<double>> load_images(const std::string &path, size_t count) { std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open MNIST images: "+path);if(read_u32(f)!=2051)throw std::runtime_error("invalid MNIST image magic");auto total=read_u32(f),rows=read_u32(f),cols=read_u32(f);if(rows!=28||cols!=28)throw std::runtime_error("MNIST images must be 28x28");size_t n=std::min<size_t>(count,total);std::vector<std::vector<double>> result(n,std::vector<double>(784));for(auto &image:result){for(double &p:image){unsigned char byte;if(!f.read((char*)&byte,1))throw std::runtime_error("truncated MNIST images");p=byte/255.0;}}return result; }
static std::vector<int> load_labels(const std::string &path, size_t count) { std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open MNIST labels: "+path);if(read_u32(f)!=2049)throw std::runtime_error("invalid MNIST label magic");auto total=read_u32(f);size_t n=std::min<size_t>(count,total);std::vector<int> result(n);for(int &label:result){unsigned char byte;if(!f.read((char*)&byte,1))throw std::runtime_error("truncated MNIST labels");label=byte;}return result; }
static int argmax(const std::vector<double> &v){return (int)std::distance(v.begin(),std::max_element(v.begin(),v.end()));}
static void print_json_string(const std::string &s){std::cout<<'"';for(char c:s){if(c=='"'||c=='\\')std::cout<<'\\';std::cout<<c;}std::cout<<'"';}

int main(int argc,char **argv){try{
    std::string images,labels;std::vector<std::string> model_paths;size_t count=10,threads=4;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--images"&&i+1<argc)images=argv[++i];else if(a=="--labels"&&i+1<argc)labels=argv[++i];else if(a=="--model"&&i+1<argc)model_paths.push_back(argv[++i]);else if(a=="--count"&&i+1<argc)count=std::stoull(argv[++i]);else if(a=="--threads"&&i+1<argc)threads=std::stoull(argv[++i]);else throw std::runtime_error("unknown argument: "+a);}
    if(images.empty()||labels.empty()||model_paths.empty()||count==0)throw std::runtime_error("usage: sealtorch_validate --images FILE --labels FILE --model FILE [--model FILE ...] --count N [--threads N]");
    auto images_data=load_images(images,count);auto labels_data=load_labels(labels,count);size_t n=std::min(images_data.size(),labels_data.size());if(!n)throw std::runtime_error("MNIST validation set is empty");
    seal::EncryptionParameters parameters(seal::scheme_type::ckks);parameters.set_poly_modulus_degree(16384);parameters.set_coeff_modulus(seal::CoeffModulus::Create(16384,{60,30,30,30,30,30,30,30,30,30,30,60}));seal::SEALContext context(parameters);seal::KeyGenerator keygen(context);seal::Encryptor encryptor(context,keygen.secret_key());seal::Decryptor decryptor(context,keygen.secret_key());seal::Evaluator evaluator(context);seal::CKKSEncoder encoder(context);seal::RelinKeys relin;seal::GaloisKeys galois;keygen.create_relin_keys(relin);keygen.create_galois_keys(galois);constexpr double scale=1073741824.0;
    std::cout<<"{\"count\":"<<n<<",\"models\":[";for(size_t m=0;m<model_paths.size();++m){if(m)std::cout<<',';auto model=load_model(model_paths[m]);sealtorch::Evaluator encrypted_model(model,threads);size_t gelu_correct=0,relu_correct=0,cipher_correct=0,agree_gelu=0,agree_relu=0;double plain_ms=0,cipher_ms=0,mean_error=0;
        for(size_t sample=0;sample<n;++sample){auto start=std::chrono::steady_clock::now();auto pg=plain_predict(model,images_data[sample],false);auto plain_end=std::chrono::steady_clock::now();auto pr=plain_predict(model,images_data[sample],true);plain_ms+=std::chrono::duration<double,std::milli>(plain_end-start).count();int pg_pred=argmax(pg),pr_pred=argmax(pr);gelu_correct+=pg_pred==labels_data[sample];relu_correct+=pr_pred==labels_data[sample];seal::Plaintext encoded;encoder.encode(images_data[sample],scale,encoded);seal::Ciphertext encrypted;encryptor.encrypt_symmetric(encoded,encrypted);std::vector<seal::Ciphertext> values{std::move(encrypted)};std::vector<double> cipher;start=std::chrono::steady_clock::now();auto output=encrypted_model.predict(values,evaluator,relin,galois,encoder,scale);cipher.reserve(output.size());for(const auto &value:output){seal::Plaintext decoded;decryptor.decrypt(value,decoded);std::vector<double> slots;encoder.decode(decoded,slots);if(slots.empty())throw std::runtime_error("empty decrypted validation output");cipher.push_back(slots.front());}auto cipher_end=std::chrono::steady_clock::now();cipher_ms+=std::chrono::duration<double,std::milli>(cipher_end-start).count();int c_pred=argmax(cipher);cipher_correct+=c_pred==labels_data[sample];agree_gelu+=c_pred==pg_pred;agree_relu+=c_pred==pr_pred;for(int j=0;j<10;++j)mean_error+=std::abs(cipher[j]-pg[j]);}
        std::cout<<"{\"name\":";print_json_string(model_paths[m]);std::cout<<",\"plain_gelu_accuracy\":"<<double(gelu_correct)/n<<",\"plain_relu_accuracy\":"<<double(relu_correct)/n<<",\"cipher_gelu_accuracy\":"<<double(cipher_correct)/n<<",\"cipher_vs_gelu_agreement\":"<<double(agree_gelu)/n<<",\"cipher_vs_relu_agreement\":"<<double(agree_relu)/n<<",\"plain_ms_per_image\":"<<plain_ms/n<<",\"cipher_ms_per_image\":"<<cipher_ms/n<<",\"mean_output_abs_error\":"<<mean_error/(n*10)<<"}";
    }std::cout<<"]}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
