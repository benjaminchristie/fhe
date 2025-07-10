#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ctime>
#include <vector>

namespace F {

typedef double (*activation_func)(double);

double identity(double x) noexcept;
std::vector<double> identity(std::vector<double> x) noexcept;
double identity_derivative(double x) noexcept;
std::vector<double> identity_derivative(std::vector<double> x) noexcept;

double ReLU(double x) noexcept;
std::vector<double> ReLU(std::vector<double> x) noexcept;
double ReLU_derivative(double x) noexcept;
std::vector<double> ReLU_derivative(std::vector<double> x) noexcept;

double sigmoid(double x) noexcept;
std::vector<double> sigmoid(std::vector<double> x) noexcept;
double sigmoid_derivative(double x) noexcept;
std::vector<double> sigmoid_derivative(std::vector<double> x) noexcept;

double tanh(double x) noexcept;
std::vector<double> tanh(std::vector<double> x) noexcept;
double tanh_derivative(double x) noexcept;
std::vector<double> tanh_derivative(std::vector<double> x) noexcept;

double random_weight() noexcept;
double random_weight(const double min, const double max) noexcept;

} // namespace F

namespace nn {

class LinearLayer {
private:
  const size_t input_size;
  const size_t output_size;
  F::activation_func f;
  std::vector<std::vector<double>> weight;
  std::vector<double> bias;

  void initialize_params();

public:
  LinearLayer(size_t input_size, size_t output_size);
  LinearLayer(size_t input_size, size_t output_size, F::activation_func f);
  ~LinearLayer();
  std::vector<double> forward(const std::vector<double> &input);
  std::vector<double> backward(const std::vector<double> &output,
                               const std::vector<double> &target,
                               const double learning_rate);
};
class MLP {
private:
public:
  MLP(size_t n_inputs, size_t n_outputs);
  ~MLP();
  std::vector<double> forward(std::vector<double> input);
  std::vector<double> backward(std::vector<double> input);
};
} // namespace nn