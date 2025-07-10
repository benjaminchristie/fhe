#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

double randWeight() {
  return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Range [-1, 1]
}

class LinearLayer {
public:
  std::vector<double> bias;
  std::vector<std::vector<double>> weight;

  std::vector<double> input;
  std::vector<double> output;
  std::vector<double> deltas;

  size_t input_size, output_size;

  LinearLayer(size_t input_dim, size_t output_dim)
      : input_size(input_dim), output_size(output_dim) {
    weight.resize(output_dim, std::vector<double>(input_dim));
    bias.resize(output_dim);
    output.resize(output_dim);
    deltas.resize(output_dim);
    input.resize(input_dim);
    for (size_t i = 0; i < output_dim; ++i) {
      bias[i] = randWeight();
      for (size_t j = 0; j < input_dim; ++j) {
        weight[i][j] = randWeight();
      }
    }
  }

  double activation_func(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
  }
  double activation_func_derivative(double x) noexcept {
    auto y = activation_func(x);
    return y * (1.0 - y);
  }

  std::vector<double> forward(const std::vector<double> &x) {
    input = x;
    for (size_t i = 0; i < output_size; i++) {
      double sum = bias[i];
      for (size_t j = 0; j < input_size; j++) {
        sum += weight[i][j] * input[j];
      }
      output[i] = activation_func(sum);
    }
    return output;
  }

  void
  computeDeltas(const std::vector<double> &target_or_next_deltas,
                const std::vector<std::vector<double>> &next_weights = {}) {
    for (size_t i = 0; i < output_size; ++i) {
      if (next_weights.empty()) {
        // Output layer
        deltas[i] = (output[i] - target_or_next_deltas[i]) *
                    activation_func_derivative(output[i]);
      } else {
        // Hidden layer
        double error = 0.0;
        for (size_t j = 0; j < target_or_next_deltas.size(); ++j) {
          error += target_or_next_deltas[j] * next_weights[j][i];
        }
        deltas[i] = error * activation_func_derivative(output[i]);
      }
    }
  }

  void updateWeights(double learning_rate) {
    for (size_t i = 0; i < output_size; ++i) {
      for (size_t j = 0; j < input_size; ++j) {
        weight[i][j] -= learning_rate * deltas[i] * input[j];
      }
      bias[i] -= learning_rate * deltas[i];
    }
  }
};

class MLP {
public:
  std::vector<LinearLayer> layers;

  MLP(const std::vector<size_t> &szs) {
    for (size_t i = 1; i < szs.size(); i++) {
      layers.emplace_back(szs[i - 1], szs[i]);
    }
  }

  std::vector<double> forward(const std::vector<double> &x) {
    std::vector<double> out = x;
    for (auto &layer : layers) {
      out = layer.forward(out);
    }
    return out;
  }

  void train(const std::vector<std::vector<double>> &X,
             const std::vector<std::vector<double>> &y, int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      double total_loss = 0.0;
      for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> out = forward(X[i]);

        // Compute loss
        for (size_t j = 0; j < out.size(); ++j) {
          total_loss += 0.5 * pow(out[j] - y[i][j], 2);
        }

        // Backpropagation
        layers.back().computeDeltas(y[i]); // Output layer

        for (int l = (int)layers.size() - 2; l >= 0; --l) {
          layers[l].computeDeltas(layers[l + 1].deltas, layers[l + 1].weight);
        }

        for (auto &layer : layers) {
          layer.updateWeights(lr);
        }
      }

      if (epoch % 100 == 0)
        std::cout << "Epoch " << epoch
                  << ", Loss: " << total_loss / (double)X.size() << std::endl;
    }
  }
};

int main(int argc, char **argv) {
  auto mlp = MLP({4, 3, 2});
  std::vector<std::vector<double>> INPUTS = {{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}};
  std::vector<std::vector<double>> OUTPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  mlp.train(INPUTS, OUTPUTS, 100000, 0.01);

  for (const auto &input : INPUTS) {
    std::vector<double> out = mlp.forward(input);
    std::cout << "Input: (" << input[0] << ", " << input[1]
         << ") => Output: " << out[0] <<" " << out[1] << std::endl;
  }

  return 0;
}