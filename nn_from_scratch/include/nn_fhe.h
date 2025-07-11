#pragma once

#include "openfhe.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ctime>
#include <memory>
#include <vector>

typedef lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<
    bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>
    FunctionalCiphertext;

namespace F_fhe {

class Functional {
public:
  virtual FunctionalCiphertext forward(FunctionalCiphertext x) noexcept = 0;
  virtual FunctionalCiphertext derivative(FunctionalCiphertext x) noexcept = 0;
  std::vector<FunctionalCiphertext>
  forward(std::vector<FunctionalCiphertext> x) noexcept;
  std::vector<FunctionalCiphertext>
  derivative(std::vector<FunctionalCiphertext> x) noexcept;
};

class Identity : public Functional {
  FunctionalCiphertext forward(FunctionalCiphertext x) noexcept;
  FunctionalCiphertext derivative(FunctionalCiphertext x) noexcept;
};

class Sigmoid : public Functional {
  FunctionalCiphertext forward(FunctionalCiphertext x) noexcept;
  FunctionalCiphertext derivative(FunctionalCiphertext x) noexcept;
};

class ReLU : public Functional {
  FunctionalCiphertext forward(FunctionalCiphertext x) noexcept;
  FunctionalCiphertext derivative(FunctionalCiphertext x) noexcept;
};

class Tanh : public Functional {
  FunctionalCiphertext forward(FunctionalCiphertext x) noexcept;
  FunctionalCiphertext derivative(FunctionalCiphertext x) noexcept;
};

double random_weight() noexcept;
double random_weight(const double min, const double max) noexcept;

std::vector<FunctionalCiphertext>
vec_dot(std::vector<FunctionalCiphertext> a,
        std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
vec_add(std::vector<FunctionalCiphertext> a,
        std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
vec_sub(std::vector<FunctionalCiphertext> a,
        std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
scalar_dot(double s, std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
scalar_div(double s, std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
scalar_add(double s, std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
scalar_sub(double s, std::vector<FunctionalCiphertext> b) noexcept;
std::vector<FunctionalCiphertext>
scalar_sub(std::vector<FunctionalCiphertext> b, double s) noexcept;

} // namespace F_fhe

namespace nn_fhe {

class LinearLayer {
private:
  const size_t input_size;
  const size_t output_size;
  std::unique_ptr<F_fhe::Functional> f;
  std::vector<std::vector<double>> weight;
  std::vector<double> bias;
  std::vector<FunctionalCiphertext> input;
  std::vector<FunctionalCiphertext> output;
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
              std::unique_ptr<F_fhe::Functional> f);
  std::vector<double> forward(const std::vector<double> &input);
  friend class MLP;
};
class MLP {
private:
  std::vector<LinearLayer> layers; // with activation

public:
  MLP(const std::vector<size_t> &sizes);
  MLP(const std::vector<size_t> &sizes,
      std::vector<std::unique_ptr<F_fhe::Functional>> &functionals);
  std::vector<FunctionalCiphertext>
  forward(const std::vector<FunctionalCiphertext> &input);
  double update(const std::vector<FunctionalCiphertext> &input,
                const std::vector<FunctionalCiphertext> &target,
                const double learning_rate);
};
} // namespace nn_fhe