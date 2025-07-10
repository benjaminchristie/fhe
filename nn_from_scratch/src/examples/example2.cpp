#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double sigmoid_derivative(double x) {
  return x * (1.0 - x); // Assumes x is already sigmoid(x)
}

double randWeight() {
  return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Range [-1, 1]
}

class Layer {
public:
  int input_size, output_size;
  vector<vector<double>> weights;
  vector<double> biases;
  vector<double> outputs;
  vector<double> inputs;
  vector<double> deltas;

  Layer(int in_size, int out_size)
      : input_size(in_size), output_size(out_size) {
    weights.resize(out_size, vector<double>(in_size));
    biases.resize(out_size);
    outputs.resize(out_size);
    deltas.resize(out_size);
    inputs.resize(in_size);
    // Random initialization
    for (int i = 0; i < out_size; ++i) {
      biases[i] = randWeight();
      for (int j = 0; j < in_size; ++j) {
        weights[i][j] = randWeight();
      }
    }
  }

  vector<double> forward(const vector<double> &input) {
    inputs = input;
    for (int i = 0; i < output_size; ++i) {
      double sum = biases[i];
      for (int j = 0; j < input_size; ++j) {
        sum += weights[i][j] * input[j];
      }
      outputs[i] = sigmoid(sum);
    }
    return outputs;
  }

  void computeDeltas(const vector<double> &target_or_next_deltas,
                     const vector<vector<double>> &next_weights = {}) {
    for (int i = 0; i < output_size; ++i) {
      if (next_weights.empty()) {
        // Output layer
        deltas[i] = (outputs[i] - target_or_next_deltas[i]) *
                    sigmoid_derivative(outputs[i]);
      } else {
        // Hidden layer
        double error = 0.0;
        for (int j = 0; j < target_or_next_deltas.size(); ++j) {
          error += target_or_next_deltas[j] * next_weights[j][i];
        }
        deltas[i] = error * sigmoid_derivative(outputs[i]);
      }
    }
  }

  void updateWeights(double learning_rate) {
    for (int i = 0; i < output_size; ++i) {
      for (int j = 0; j < input_size; ++j) {
        weights[i][j] -= learning_rate * deltas[i] * inputs[j];
      }
      biases[i] -= learning_rate * deltas[i];
    }
  }
};

class NeuralNetwork {
public:
  vector<Layer> layers;

  NeuralNetwork(const vector<int> &layer_sizes) {
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
      layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
    }
  }

  vector<double> predict(const vector<double> &input) {
    vector<double> out = input;
    for (auto &layer : layers) {
      out = layer.forward(out);
    }
    return out;
  }

  void train(const vector<vector<double>> &X, const vector<vector<double>> &y,
             int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      double total_loss = 0.0;
      for (size_t i = 0; i < X.size(); ++i) {
        vector<double> out = predict(X[i]);

        // Compute loss
        for (size_t j = 0; j < out.size(); ++j) {
          total_loss += 0.5 * pow(out[j] - y[i][j], 2);
        }

        // Backpropagation
        layers.back().computeDeltas(y[i]); // Output layer

        for (int l = layers.size() - 2; l >= 0; --l) {
          layers[l].computeDeltas(layers[l + 1].deltas, layers[l + 1].weights);
        }

        for (auto &layer : layers) {
          layer.updateWeights(lr);
        }
      }

      if (epoch % 100 == 0)
        cout << "Epoch " << epoch << ", Loss: " << total_loss / X.size()
             << endl;
    }
  }
};
int main() {
  srand(time(0));

  // Example: 2 inputs → 2 hidden → 1 output
  NeuralNetwork nn({2, 3, 2});

  vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  vector<vector<double>> y = {{0, 0}, {1, 1}, {0, 0}, {1, 1}};

  nn.train(X, y, 10000, 0.5);

  for (const auto &input : X) {
    vector<double> out = nn.predict(input);
    cout << "Input: (" << input[0] << ", " << input[1]
         << ") => Output: " << out[0] << " " << out[1] << endl;
  }

  return 0;
}
