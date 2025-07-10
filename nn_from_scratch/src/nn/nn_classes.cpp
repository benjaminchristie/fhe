#include "nn.h"

namespace nn {

LinearLayer::LinearLayer(size_t input_size, size_t output_size)
    : input_size(input_size), output_size(output_size),
      _F(std::make_unique<F::Identity>()) {
  initialize_params();
}

LinearLayer::LinearLayer(size_t input_size, size_t output_size,
                         std::unique_ptr<F::Functional> &fu)
    : input_size(input_size), output_size(output_size) {
  _F = std::move(fu);
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
  deltas = std::vector<double>(output_size, 0.0);
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
    result[j] = _F->forward(sum);
  }
  return result;
}

MLP::MLP(std::vector<size_t> structure,
         std::vector<std::unique_ptr<F::Functional>> &activations) {
  size_t sz = structure.size();
  assert(sz == activations.size() + 1);
  // layers.reserve(sz - 1);
  for (size_t i = 0; i < sz - 1; i++) {
    layers.emplace_back(std::make_unique<LinearLayer>(
        structure[i], structure[i + 1], activations[i]));
  }
}

MLP::~MLP() {}

std::vector<double> MLP::forward(const std::vector<double> &input) {
  std::vector<double> output = input;
  for (size_t i = 0; i < layers.size(); i++) {
    output = layers[i]->forward(output);
  }
  return output;
}

/*
returns a vector of column vectors organized as follows:
[ a1 b1 a2 b2 ... ]
 where a represents *before* activation is applied and b is after
 for each layer in the network
*/
std::vector<std::vector<double>> MLP::forward(const std::vector<double> &input,
                                              std::vector<double> &output) {
  std::vector<std::vector<double>> v(layers.size() * 2 + 1);
  output = input;
  v[0] = std::vector<double>(input.size());
  std::copy(input.begin(), input.end(), v[0].begin());
  for (size_t i = 0; i < layers.size(); i++) {
    auto output_size = layers[i]->output_size;
    auto input_size = layers[i]->input_size;
    std::vector<double> result(output_size);

    v[2 * i + 1] = std::vector<double>(output_size);
    v[2 * i + 2] = std::vector<double>(output_size);

    for (size_t j = 0; j < output_size; j++) {
      double s = layers[i]->bias[j];
      for (size_t k = 0; k < input_size; k++) {
        s += output[i] * layers[i]->weight[i][j];
      }
      v[2 * i + 1][j] = s;
      v[2 * i + 2][j] = layers[i]->_F->forward(s);
    }
    output = v[2 * i + 1];
  }
  return v;
}

/*
updates the parameters of linear layers within the network

note for a single weight u_ij, the partial derivative of the
loss function w.r.t. u_ij is:

dL       dL    dyj    daj^2
--    =  --  * ---  * -----
du_ij    dyj  daj^2   du_ij

where i corrspon

*/
void MLP::backward(const std::vector<double> &input,
                   const std::vector<double> &target,
                   const double learning_rate) {
  std::vector<double> output;
  auto all_outputs =
      forward(input, output); // 2 * n_layers + 1 vector of vectors
  auto dZ = F::scalar_dot(2.0, F::vec_sub(output, target));
  int n_layers = (int)layers.size();
  // backprop hidden layers
  _compute_deltas(layers[n_layers - 1], layers[n_layers - 1], output, target,
                  true);
  for (int l = (int)n_layers - 2; l >= 0; l--) {
    _compute_deltas(layers[l], layers[l + 1], all_outputs[2 * l + 2],
                    layers[l + 1]->deltas, false);
  }
  for (int l = 0; l < n_layers; l++) {
    auto &layer = layers[l];
    auto &_input = all_outputs[2 * l];
    for (size_t i = 0; i < layer->output_size; i++) {
      for (size_t j = 0; j < layer->input_size; j++) {
        layer->weight[i][j] -= learning_rate * layer->deltas[i] * _input[j];
      }
      layer->bias[i] -= learning_rate * layer->deltas[i];
    }
  }
}

void MLP::_compute_deltas(std::unique_ptr<LinearLayer> &l1,
                          std::unique_ptr<LinearLayer> &l2,
                          const std::vector<double> &outputs,
                          const std::vector<double> &target, bool isFinal) {
  size_t output_size = l1->output_size;
  if (isFinal) {
    for (size_t i = 0; i < output_size; i++) {
      l1->deltas[i] = (outputs[i] - target[i]) * l1->_F->derivative(outputs[i]);
    }
  } else {
    for (size_t i = 0; i < output_size; i++) {
      double error = 0.0;
      for (size_t j = 0; j < target.size(); j++) {
        error += target[j] * l2->weight[j][i];
      }
      l1->deltas[i] = error * l1->_F->derivative(outputs[i]);
    }
  }
}

} // namespace nn