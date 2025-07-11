#include "nn_fhe.h"

using namespace lbcrypto;

namespace nn_fhe {

LinearLayer::LinearLayer(size_t input_size, size_t output_size,
                         std::unique_ptr<F_fhe::Functional> fu,
                         CryptoContext<DCRTPoly> cc)
    : LinearLayer::LinearLayer(input_size, output_size, cc) {
  f = std::move(fu);
  f->cc = cc;
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size,
                         CryptoContext<DCRTPoly> cc)
    : input_size(input_size), output_size(output_size), cc(cc) {
  weight.resize(output_size, std::vector<double>(input_size));
  bias.resize(output_size);
  output.resize(output_size);
  input.resize(input_size);
  f = std::make_unique<F_fhe::Identity>();
  f->cc = cc;
}

std::vector<FunctionalCiphertext>
LinearLayer::forward(const std::vector<FunctionalCiphertext> &x) {
  input = x;
  for (size_t i = 0; i < output_size; i++) {
    auto zero = cc->EvalSub(input[i], input[i]);
    auto sum = cc->EvalAdd(zero, bias[i]);
    for (size_t j = 0; j < input_size; j++) {
      auto curr = cc->EvalMult(input[j], weight[i][j]);
      sum = cc->EvalAdd(curr, sum);
    }
    output[i] = f->forward(sum);
  }
  return output;
}

MLP::MLP(const std::vector<size_t> &sizes, CryptoContext<DCRTPoly> cc) {
  for (size_t i = 1; i < sizes.size(); i++) {
    layers.emplace_back(sizes[i - 1], sizes[i],
                        std::make_unique<F_fhe::Identity>(), cc);
  }
}

MLP::MLP(const std::vector<size_t> &sizes,
         std::vector<std::unique_ptr<F_fhe::Functional>> &functionals,
         CryptoContext<DCRTPoly> cc) {
  assert(sizes.size() == functionals.size() + 1);
  for (size_t i = 1; i < sizes.size(); i++) {
    layers.emplace_back(sizes[i - 1], sizes[i], std::move(functionals[i - 1]),
                        cc);
  }
}
std::vector<FunctionalCiphertext>
MLP::forward(const std::vector<FunctionalCiphertext> &x) {
  std::vector<FunctionalCiphertext> out = x;
  for (auto &layer : layers) {
    out = layer.forward(out);
  }
  return out;
}

void MLP::set_parameters(nn::MLP &source) {
  auto [W, B] = source.get_parameters();
  for (size_t i = 0; i < W.size(); i++) {
    layers[i].weight = W[i];
    layers[i].bias = B[i];
  }
}

} // namespace nn_fhe