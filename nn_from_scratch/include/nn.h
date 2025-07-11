#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ctime>
#include <memory>
#include <vector>

namespace F {

class Functional {
public:
  virtual double forward(double x) noexcept = 0;
  virtual double derivative(double x) noexcept = 0;
  std::vector<double> forward(std::vector<double> x) noexcept;
  std::vector<double> derivative(std::vector<double> x) noexcept;
};

class Identity : public Functional {
  double forward(double x) noexcept;
  double derivative(double x) noexcept;
};

class Sigmoid : public Functional {
  double forward(double x) noexcept;
  double derivative(double x) noexcept;
};

class ReLU : public Functional {
  double forward(double x) noexcept;
  double derivative(double x) noexcept;
};

class Tanh : public Functional {
  double forward(double x) noexcept;
  double derivative(double x) noexcept;
};

double random_weight() noexcept;
double random_weight(const double min, const double max) noexcept;

std::vector<double> vec_dot(std::vector<double> a,
                            std::vector<double> b) noexcept;
std::vector<double> vec_add(std::vector<double> a,
                            std::vector<double> b) noexcept;
std::vector<double> vec_sub(std::vector<double> a,
                            std::vector<double> b) noexcept;
std::vector<double> scalar_dot(double s, std::vector<double> b) noexcept;
std::vector<double> scalar_div(double s, std::vector<double> b) noexcept;
std::vector<double> scalar_add(double s, std::vector<double> b) noexcept;
std::vector<double> scalar_sub(double s, std::vector<double> b) noexcept;
std::vector<double> scalar_sub(std::vector<double> b, double s) noexcept;

} // namespace F

// namespace Optim {
// class Optimizer {
//   virtual void
//   compute_deltas(const std::vector<double> &target_deltas,
//                  const std::vector<std::vector<double>> &next_weights) = 0;
//   virtual void update_weights(double learning_rate) = 0;
// };
// } // namespace Optim

namespace nn {

class LinearLayer {
private:
  const size_t input_size;
  const size_t output_size;
  std::unique_ptr<F::Functional> f;
  std::vector<std::vector<double>> weight;
  std::vector<double> bias;
  std::vector<double> input;
  std::vector<double> output;
  std::vector<double> deltas;

public:
  LinearLayer(LinearLayer &&other) noexcept = default;
  LinearLayer &operator=(LinearLayer &&other) noexcept = delete;

  // LinearLayer is not copyable (implicitly deleted due to unique_ptr member)
  LinearLayer(const LinearLayer &other) = delete;
  LinearLayer &operator=(const LinearLayer &other) = delete;

  void initialize_params();
  void compute_deltas(const std::vector<double> &target_deltas,
                      const std::vector<std::vector<double>> &next_weights);
  void update_weights(double learning_rate);

  LinearLayer(size_t input_size, size_t output_size);
  LinearLayer(size_t input_size, size_t output_size,
              std::unique_ptr<F::Functional> f);
  std::vector<double> forward(const std::vector<double> &input);
  friend class MLP;
  // friend class Optim::Optimizer;
};
class MLP {
private:
  std::vector<LinearLayer> layers; // with activation

public:
  MLP(const std::vector<size_t> &sizes);
  MLP(const std::vector<size_t> &sizes,
      std::vector<std::unique_ptr<F::Functional>> &functionals);
  std::vector<double> forward(const std::vector<double> &input);
  double update(const std::vector<double> &input,
                const std::vector<double> &target, const double learning_rate);
};
} // namespace nn