#include "nn_fhe.h"

namespace nn_fhe {

LinearLayer::LinearLayer(size_t input_size, size_t output_size,
                         std::unique_ptr<F_fhe::Functional> fu)
    : LinearLayer::LinearLayer(input_size, output_size) {
  f = std::move(fu);
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size) {
  weight.resize(output_size, std::vector<double>(input_size));
  bias.resize(output_size);
  output.resize(output_size);
  deltas.resize(output_size);
  input.resize(input_size);
  for (size_t i = 0; i < output_size; i++) {
    bias[i] = F_fhe::random_weight();
    for (size_t j = 0; j < input_size; j++) {
      weight[i][j] = F_fhe::random_weight();
    }
  }
  f = std::make_unique<F_fhe::Identity>();
}

std::vector<double> LinearLayer::forward(const std::vector<double> &x) {
  input = x;
  for (size_t i = 0; i < output_size; i++) {
    double sum = bias[i];
    for (size_t j = 0; j < input_size; j++) {
      sum += weight[i][j] * input[j];
    }
    output[i] = f->forward(sum);
  }
  return output;
}

void LinearLayer::compute_deltas(
    const std::vector<double> &target_or_next_deltas,
    const std::vector<std::vector<double>> &next_weights = {}) {
  if (next_weights.empty()) { // output layer
    for (size_t i = 0; i < output_size; ++i) {
      deltas[i] =
          (output[i] - target_or_next_deltas[i]) * f->derivative(output[i]);
    }
  } else { // hidden layer
    for (size_t i = 0; i < output_size; ++i) {
      double error = 0.0;
      for (size_t j = 0; j < target_or_next_deltas.size(); ++j) {
        error += target_or_next_deltas[j] * next_weights[j][i];
      }
      deltas[i] = error * f->derivative(output[i]);
    }
  }
}

void LinearLayer::update_weights(double learning_rate) {
  for (size_t i = 0; i < output_size; ++i) {
    for (size_t j = 0; j < input_size; ++j) {
      weight[i][j] -= learning_rate * deltas[i] * input[j];
    }
    bias[i] -= learning_rate * deltas[i];
  }
}

MLP::MLP(const std::vector<size_t> &sizes) {
  for (size_t i = 1; i < sizes.size(); i++) {
    layers.emplace_back(sizes[i - 1], sizes[i],
                        std::make_unique<F::Identity>());
  }
}

MLP::MLP(const std::vector<size_t> &sizes,
         std::vector<std::unique_ptr<F::Functional>> &functionals) {
  assert(sizes.size() == functionals.size() + 1);
  for (size_t i = 1; i < sizes.size(); i++) {
    layers.emplace_back(sizes[i - 1], sizes[i], std::move(functionals[i - 1]));
  }
}
std::vector<double> MLP::forward(const std::vector<double> &x) {
  std::vector<double> out = x;
  for (auto &layer : layers) {
    out = layer.forward(out);
  }
  return out;
}
double MLP::update(const std::vector<double> &input,
                   const std::vector<double> &target,
                   const double learning_rate) {
  double total_loss = 0.0;
  auto out = forward(input);
  for (size_t j = 0; j < out.size(); j++) {
    total_loss += 0.5 * std::pow(out[j] - target[j], 2);
  }
  // backprop
  layers.back().compute_deltas(target);
  for (int l = (int)layers.size() - 2; l >= 0; l--) {
    layers[l].compute_deltas(layers[l + 1].deltas, layers[l + 1].weight);
  }
  for (auto &layer : layers) {
    layer.update_weights(learning_rate);
  }
  return total_loss;
}

} // namespace nn_fhe