#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

using Vec = vector<double>;
using Matrix = vector<Vec>;

double randWeight() {
  static random_device rd;
  static mt19937 gen(rd());
  static uniform_real_distribution<> dis(-1.0, 1.0);
  return dis(gen);
}

Vec relu(const Vec &v) {
  Vec out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = max(0.0, v[i]);
  return out;
}

Vec relu_deriv(const Vec &v) {
  Vec out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = v[i] > 0 ? 1.0 : 0.0;
  return out;
}

Vec softmax(const Vec &v) {
  Vec out(v.size());
  double maxVal = *max_element(v.begin(), v.end());
  double sum = 0.0;
  for (double val : v)
    sum += exp(val - maxVal);
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = exp(v[i] - maxVal) / sum;
  return out;
}

Vec mse_loss_deriv(const Vec &output, const Vec &target) {
  Vec grad(output.size());
  for (size_t i = 0; i < output.size(); ++i)
    grad[i] = output[i] - target[i];
  return grad;
}

class Layer {
public:
  Matrix weights;
  Vec biases;
  Vec input, output;

  Layer(int in_size, int out_size) {
    weights = Matrix(out_size, Vec(in_size));
    biases = Vec(out_size);
    for (auto &row : weights)
      for (auto &w : row)
        w = randWeight();
    for (auto &b : biases)
      b = randWeight();
  }

  Vec forward(const Vec &in) {
    input = in;
    output = Vec(biases);
    for (size_t i = 0; i < weights.size(); ++i)
      for (size_t j = 0; j < in.size(); ++j)
        output[i] += weights[i][j] * in[j];
    return output;
  }
};

class NeuralNet {
public:
  vector<Layer> layers;

  NeuralNet(const vector<int> &sizes) {
    for (size_t i = 0; i < sizes.size() - 1; ++i)
      layers.emplace_back(sizes[i], sizes[i + 1]);
  }

  Vec forward(const Vec &x) {
    Vec a = x;
    for (size_t i = 0; i < layers.size(); ++i) {
      a = layers[i].forward(a);
      if (i != layers.size() - 1)
        a = relu(a);
    }
    return softmax(a);
  }

  void backward(const Vec &target, double lr) {
    Vec delta = mse_loss_deriv(layers.back().output, target);

    for (int l = layers.size() - 1; l >= 0; --l) {
      Layer &layer = layers[l];
      Vec prev_output = (l == 0) ? layer.input : relu(layers[l - 1].output);
      Vec grad = (l == layers.size() - 1) ? delta : relu_deriv(layer.output);

      Vec new_delta(layer.weights[0].size(), 0.0);
      for (size_t i = 0; i < layer.weights.size(); ++i) {
        for (size_t j = 0; j < layer.weights[i].size(); ++j) {
          double d = grad[i] * delta[i] * prev_output[j];
          layer.weights[i][j] -= lr * d;
          new_delta[j] += layer.weights[i][j] * grad[i] * delta[i];
        }
        layer.biases[i] -= lr * grad[i] * delta[i];
      }
      delta = new_delta;
    }
  }
};

int main() {
  NeuralNet net({2, 8, 6, 4, 3}); // 2 input, 3 output, 3 hidden layers

  // Sample XOR-like inputs
  vector<Vec> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  // Target: one-hot outputs for 3 classes
  vector<Vec> targets = {{1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}};

  // Training loop
  for (int epoch = 0; epoch < 1000; ++epoch) {
    double loss = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      Vec out = net.forward(inputs[i]);
      net.backward(targets[i], 0.05);

      for (size_t j = 0; j < out.size(); ++j)
        loss += pow(out[j] - targets[i][j], 2);
    }
    if (epoch % 100 == 0)
      cout << "Epoch " << epoch << ", Loss: " << loss << endl;
  }

  // Testing
  for (const auto &input : inputs) {
    Vec out = net.forward(input);
    cout << "Input: ";
    for (double x : input)
      cout << x << " ";
    cout << " -> Output: ";
    for (double o : out)
      cout << o << " ";
    cout << endl;
  }

  return 0;
}
