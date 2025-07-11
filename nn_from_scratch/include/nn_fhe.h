#pragma once

#include "nn.h"
#include "openfhe.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ctime>
#include <memory>
#include <vector>

// see https://openreview.net/pdf?id=rkxsgkHKvH

typedef lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<
    bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>
    FunctionalCiphertext;

namespace F_fhe {

class Functional {
public:
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
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
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;

public:
  // LinearLayer is not copyable (implicitly deleted due to unique_ptr member)
  LinearLayer(LinearLayer &&other) noexcept = default;
  LinearLayer &operator=(LinearLayer &&other) noexcept = delete;
  LinearLayer(const LinearLayer &other) = delete;
  LinearLayer &operator=(const LinearLayer &other) = delete;

  LinearLayer(size_t input_size, size_t output_size,
              lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);
  LinearLayer(size_t input_size, size_t output_size,
              std::unique_ptr<F_fhe::Functional> f,
              lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);
  std::vector<FunctionalCiphertext>
  forward(const std::vector<FunctionalCiphertext> &input);
  friend class MLP;
};
class MLP {
private:
  std::vector<LinearLayer> layers; // with activation

public:
  MLP(const std::vector<size_t> &sizes,
      lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);
  MLP(const std::vector<size_t> &sizes,
      std::vector<std::unique_ptr<F_fhe::Functional>> &functionals,
      lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);
  std::vector<FunctionalCiphertext>
  forward(const std::vector<FunctionalCiphertext> &input);
  void set_parameters(nn::MLP &source);
};
} // namespace nn_fhe