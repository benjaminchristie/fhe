#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Mat;

// Utility functions
double random_weight() {
  return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
}

Vec tanh_vec(const Vec &v) {
  Vec out(v.size());
  for (size_t i = 0; i < v.size(); i++)
    out[i] = tanh(v[i]);
  return out;
}

Vec tanh_derivative(const Vec &v) {
  Vec out(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    double t = tanh(v[i]);
    out[i] = 1 - t * t;
  }
  return out;
}

Vec vec_add(const Vec &a, const Vec &b) {
  Vec out(a.size());
  for (size_t i = 0; i < a.size(); i++)
    out[i] = a[i] + b[i];
  return out;
}

Vec vec_sub(const Vec &a, const Vec &b) {
  Vec out(a.size());
  for (size_t i = 0; i < a.size(); i++)
    out[i] = a[i] - b[i];
  return out;
}

Vec vec_mul(const Vec &a, const Vec &b) {
  Vec out(a.size());
  for (size_t i = 0; i < a.size(); i++)
    out[i] = a[i] * b[i];
  return out;
}

Vec scalar_mul(const Vec &a, double s) {
  Vec out(a.size());
  for (size_t i = 0; i < a.size(); i++)
    out[i] = a[i] * s;
  return out;
}

Vec mat_vec_mul(const Mat &m, const Vec &v) {
  Vec out(m.size());
  for (size_t i = 0; i < m.size(); i++) {
    out[i] = 0;
    for (size_t j = 0; j < v.size(); j++) {
      out[i] += m[i][j] * v[j];
    }
  }
  return out;
}

Mat init_weights(int rows, int cols) {
  Mat w(rows, Vec(cols));
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      w[i][j] = random_weight();
  return w;
}

// Neural network
class NeuralNet {
public:
  int input_size, hidden_size, output_size;
  Mat W1, W2;
  Vec b1, b2;
  double learning_rate;

  NeuralNet(int input, int hidden, int output, double lr = 0.01) {
    input_size = input;
    hidden_size = hidden;
    output_size = output;
    learning_rate = lr;

    W1 = init_weights(hidden, input);
    b1 = Vec(hidden, 0.0);
    W2 = init_weights(output, hidden);
    b2 = Vec(output, 0.0);
  }

  Vec forward(const Vec &x, Vec &z1, Vec &a1, Vec &z2) {
    z1 = vec_add(mat_vec_mul(W1, x), b1);
    a1 = tanh_vec(z1);
    z2 = vec_add(mat_vec_mul(W2, a1), b2);
    return z2; // No activation on output
  }

  double compute_loss(const Vec &y_true, const Vec &y_pred) {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); i++)
      loss += pow(y_true[i] - y_pred[i], 2);
    return loss;
  }

  void backward(const Vec &x, const Vec &y_true) {
    Vec z1, a1, z2;
    Vec y_pred = forward(x, z1, a1, z2);

    // Compute output layer gradient: dL/dz2 = 2 * (y_pred - y_true)
    Vec dz2 = scalar_mul(vec_sub(y_pred, y_true), 2.0);

    // Gradients for W2 and b2
    Mat dW2(output_size, Vec(hidden_size));
    Vec db2 = dz2;
    for (int i = 0; i < output_size; i++)
      for (int j = 0; j < hidden_size; j++)
        dW2[i][j] = dz2[i] * a1[j];

    // Backprop to hidden layer
    Vec dz1(hidden_size, 0.0);
    Vec tanh_deriv = tanh_derivative(z1);
    for (int j = 0; j < hidden_size; j++) {
      for (int k = 0; k < output_size; k++) {
        dz1[j] += dz2[k] * W2[k][j];
      }
      dz1[j] *= tanh_deriv[j];
    }

    // Gradients for W1 and b1
    Mat dW1(hidden_size, Vec(input_size));
    Vec db1 = dz1;
    for (int i = 0; i < hidden_size; i++)
      for (int j = 0; j < input_size; j++)
        dW1[i][j] = dz1[i] * x[j];

    // Gradient descent step
    for (int i = 0; i < output_size; i++)
      for (int j = 0; j < hidden_size; j++)
        W2[i][j] -= learning_rate * dW2[i][j];
    for (int i = 0; i < output_size; i++)
      b2[i] -= learning_rate * db2[i];

    for (int i = 0; i < hidden_size; i++)
      for (int j = 0; j < input_size; j++)
        W1[i][j] -= learning_rate * dW1[i][j];
    for (int i = 0; i < hidden_size; i++)
      b1[i] -= learning_rate * db1[i];
  }
};

int main() {
  srand(time(0));

  NeuralNet net(3, 5, 2, 0.01); // 3 input, 5 hidden, 2 output

  // Dummy training data
  vector<Vec> X = {{0.0, 0.0, 0.0}, {1.0, 0.5, -1.0}, {0.3, -0.2, 0.9}};
  vector<Vec> Y = {{0.0, 0.0}, {1.0, -1.0}, {0.5, 0.7}};

  for (int epoch = 0; epoch < 1000; epoch++) {
    double total_loss = 0.0;
    for (size_t i = 0; i < X.size(); i++) {
      net.backward(X[i], Y[i]);
      Vec z1, a1, z2;
      Vec pred = net.forward(X[i], z1, a1, z2);
      total_loss += net.compute_loss(Y[i], pred);
    }
    if (epoch % 100 == 0)
      cout << "Epoch " << epoch << " Loss: " << total_loss << endl;
  }

  // Test
  Vec z1, a1, z2;
  Vec result = net.forward({1.0, 0.5, -1.0}, z1, a1, z2);
  cout << "Prediction: ";
  for (double val : result)
    cout << val << " ";
  cout << endl;

  return 0;
}
