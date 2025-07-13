#include "nn_pure.h"
#include <cstddef>

namespace utils {
std::vector<double> vec_add(const std::vector<double>& a,
                            const std::vector<double>& b) noexcept {
    auto sz = a.size();
    assert(sz == b.size());
    std::vector<double> output(sz);
    std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                   std::plus<double>());
    return output;
}
std::vector<double> vec_add(const std::vector<double>& a, double b) noexcept {
    auto output = a;
    std::transform(a.begin(), a.end(), output.begin(),
                   [b](double d) { return d + b; });
    return output;
}

std::vector<double> vec_sub(const std::vector<double>& a,
                            const std::vector<double>& b) noexcept {
    auto sz = a.size();
    assert(sz == b.size());
    std::vector<double> output(sz);
    std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                   std::minus<double>());
    return output;
}
std::vector<double> vec_sub(const std::vector<double>& a, double b) noexcept {
    auto output = a;
    std::transform(a.begin(), a.end(), output.begin(),
                   [b](double d) { return d - b; });
    return output;
}
std::vector<double> vec_sub(double a, const std::vector<double>& b) noexcept {
    auto output = b;
    std::transform(b.begin(), b.end(), output.begin(),
                   [a](double d) { return a - d; });
    return output;
}

std::vector<double> vec_mult(const std::vector<double>& a,
                             const std::vector<double>& b) noexcept {
    auto sz = a.size();
    assert(sz == b.size());
    std::vector<double> output(sz);
    std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                   [](double x, double y) { return x * y; });
    return output;
}
std::vector<double> vec_mult(double a, const std::vector<double>& b) noexcept {
    auto output = b;
    std::transform(b.begin(), b.end(), output.begin(),
                   [a](double d) { return a * d; });
    return output;
}

double vec_dot(const std::vector<double>& a,
               const std::vector<double>& b) noexcept {
    assert(a.size() == b.size());
    double output = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    return output;
}

double rand_double(double min, double max) noexcept {
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}

double rand_double() noexcept { return rand_double(-1.0, 1.0); }

} // namespace utils

/*
Note that all of the following functions use
polynomial approximations for their activations
*/

namespace F {

double tanh(double x) noexcept {
    auto x3 = std::pow(x, 3);
    auto x5 = std::pow(x, 5);
    return x + x3 * (-1.0 / 3.0) + x5 * (2.0 / 15.0);
}

double tanh_derivative(double x) noexcept {
    double x2 = std::pow(x, 2);
    double x4 = std::pow(x, 4);
    return 1 - x2 + (2.0 / 3.0) * x4;
}

double sigmoid(double x) noexcept {
    auto x3 = std::pow(x, 3);
    auto x5 = std::pow(x, 5);
    return 0.5 + x / 4.0 - x3 / 48.0 + x5 / 480.0;
}

double sigmoid_derivative(double x) noexcept {
    double x2 = std::pow(x, 2);
    double x4 = std::pow(x, 4);
    return 1.0 / 4.0 - x2 / (48.0 / 3.0) + x4 / (480.0 / 5.0);
}

double identity(double x) noexcept { return x; }
double identity_derivative(double x) noexcept {
    (void)x; // remove unused parameter warnings;
    return 1.0;
}

} // namespace F

namespace NN {
LinearLayer::LinearLayer(size_t input_size, size_t output_size, F::Activation f)
    : activation(f), derivative([&] {
          if (f == F::tanh) {
              return F::tanh_derivative;
          } else if (f == F::sigmoid) {
              return F::sigmoid_derivative;
          } else if (f == F::identity) {
              return F::identity_derivative;
          } else {
              return F::identity_derivative; // i will explode!
          }
      }()),
      input_size(input_size), output_size(output_size) {
    initialize_params();
}

void LinearLayer::initialize_params() noexcept {
    biases = std::vector<double>(output_size);
    weights = std::vector<std::vector<double>>(output_size,
                                               std::vector<double>(input_size));
    // DO_PARALLEL
    for (size_t i = 0; i < output_size; i++) {
        biases[i] = utils::rand_double(); // between -1 and 1;
        for (size_t j = 0; j < input_size; j++) {
            weights[i][j] = utils::rand_double();
        }
    }
    deltas.resize(output_size);
    input.resize(input_size);
    output.resize(output_size);
}

std::vector<double> LinearLayer::forward(const std::vector<double>& x) {
    input = x;
    // DO_PARALLEL
    for (size_t i = 0; i < output_size; i++) {
        double dot_product = utils::vec_dot(weights[i], input);
        output[i] = activation(dot_product + biases[i]);
    }
    return output;
}

void LinearLayer::_forward(const std::vector<double>& x,
                           std::vector<double>& result) {
    input = x;
    result.resize(output_size);
    // DO_PARALLEL
    for (size_t i = 0; i < output_size; i++) {
        double dot_product = utils::vec_dot(weights[i], input);
        result[i] = activation(dot_product + biases[i]);
    }
    output = result;
}

void LinearLayer::compute_deltas(
    const std::vector<double>& target_deltas,
    const std::vector<std::vector<double>>& next_weights = {}) {
    // hidden layer
    if (!next_weights.empty()) {
        // DO_PARALLEL
        for (size_t i = 0; i < output_size; i++) {
            double error = 0.0;
            // this next part is potentially very slow
            // since the next_weights[j] access will
            // likely result in a cache miss
            for (size_t j = 0; j < target_deltas.size(); j++) {
                error += target_deltas[j] * next_weights[j][i];
            }
            deltas[i] = error * derivative(output[i]);
        }
    }
    // output layer
    else {
        // DO_PARALLEL
        for (size_t i = 0; i < output_size; i++) {
            deltas[i] = (output[i] - target_deltas[i]) * derivative(output[i]);
        }
    }
}

void LinearLayer::update_weights(double learning_rate) {
    // NEW:
    biases = utils::vec_sub(biases, utils::vec_mult(learning_rate, deltas));
    // DO_PARALLEL
    for (size_t i = 0; i < output_size; i++) {
        weights[i] = utils::vec_sub(
            weights[i], utils::vec_mult(learning_rate * deltas[i], input));
    }
    // OLD:
    /*
    for (size_t i = 0; i < output_size; i++) {
        for (size_t j = 0; j < input_size; j++) {
            weights[i][j] -= learning_rate * deltas[i] * input[j];
        }
        biases[i] -= learning_rate * deltas[i];
    }
    */
}

MLP::MLP(const std::vector<size_t>& szs) {
    for (size_t i = 0; i < szs.size() - 1; i++) {
        layers.emplace_back(szs[i], szs[i + 1], F::identity);
    }
}

MLP::MLP(const std::vector<size_t>& szs, const std::vector<F::Activation>& fs) {
    assert(fs.size() + 1 == szs.size());
    for (size_t i = 0; i < szs.size() - 1; i++) {
        layers.emplace_back(szs[i], szs[i + 1], fs[i]);
    }
}

std::vector<double> MLP::forward(const std::vector<double>& x) {
    std::vector<double> _x(x.size());
    _forward(x, _x);
    return _x;
}

void MLP::_forward(const std::vector<double>& x, std::vector<double>& result) {
    auto _x = x;
    for (auto& layer : layers) {
        layer._forward(_x, result);
        _x = result;
    }
}

double MLP::update(const std::vector<double>& input,
                   const std::vector<double>& target,
                   const double learning_rate) {
    // auto pred = forward(input);
    // auto diff = utils::vec_sub(pred, target);
    // double loss = utils::vec_dot(diff, diff);
    double loss = 0.0;
    layers.back().compute_deltas(target);
    for (int l = layers.size() - 2; l >= 0; l--) {
        layers[l].compute_deltas(layers[l + 1].deltas, layers[l + 1].weights);
    }
    for (auto& layer : layers) {
        layer.update_weights(learning_rate);
    }
    return loss;
}

void MLP::train(const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>> Y, const int epochs,
                const double learning_rate, bool verbose = true) {
    assert(X.size() == Y.size());
    auto sz = X.size();
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < sz; i++) {
            update(X[i], Y[i], learning_rate);
        }
        if (verbose && epoch % (epochs / 10) == 0) {
            std::cout << "On epoch " << epoch << "\n";
        }
    }
}

} // namespace NN