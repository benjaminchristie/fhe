#include "nn.h"

namespace nn {

LinearLayer::LinearLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size), f(F::identity) {
  initialize_params();
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size,
                         F::activation_func f)
    : input_size(input_size), output_size(output_size), f(f) {
  initialize_params();
}

LinearLayer::~LinearLayer() {}

void LinearLayer::initialize_params() {
  weight = std::vector<std::vector<double>>(input_size,
                                            std::vector<double>(output_size));
  bias = std::vector<double>(output_size);
  for (size_t i = 0; i < output_size; i++) {
    bias[i] = F::random_weight();
    for (size_t j = 0; j < input_size; j++) {
      weight[j][i] = F::random_weight();
    }
  }
}

// naive implementation for now
std::vector<double> LinearLayer::forward(const std::vector<double> &input) {
  assert(input.size() == input_size);
  std::vector<double> result(output_size);
  for (size_t j = 0; j < output_size; j++) {
    double sum = bias[j];
    for (size_t i = 0; i < input_size; i++) {
      sum += input[i] * weight[i][j];
    }
    result[j] = f(sum);
  }
  return result;
}

/*
perform backpropogation. naive implementation
*/
std::vector<double> LinearLayer::backward(const std::vector<double> &output,
                                          const std::vector<double> &target,
                                          const double learning_rate) {
  std::vector<double> errors(output_size, 0.0);
  std::vector<double> grads(input_size, 0.0);
  for (size_t k = 0; k < output_size; k++) {
    errors[k] = output[k] - target[k];
  }
}

} // namespace nn