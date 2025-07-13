#pragma once

#include "openfhe.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#ifdef NN_USE_OMP_PARALLEL
#include <omp.h>
#define DO_PARALLEL _Pragma("omp parallel for")
#else
#define DO_PARALLEL ;
#endif

namespace utils {

std::vector<double> vec_add(const std::vector<double>& a,
                            const std::vector<double>& b) noexcept;
std::vector<double> vec_add(const std::vector<double>& a, double b) noexcept;

std::vector<double> vec_sub(const std::vector<double>& a,
                            const std::vector<double>& b) noexcept;
std::vector<double> vec_sub(const std::vector<double>& a, double b) noexcept;
std::vector<double> vec_sub(double a, const std::vector<double>& b) noexcept;

std::vector<double> vec_mult(const std::vector<double>& a,
                             const std::vector<double>& b) noexcept;
std::vector<double> vec_mult(double a, const std::vector<double>& b) noexcept;

double vec_dot(const std::vector<double>& a,
               const std::vector<double>& b) noexcept;

double rand_double(double min, double max) noexcept;
double rand_double() noexcept;

} // namespace utils

namespace F {

typedef double (*Activation)(double);

typedef lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<
    bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>
    FCiphertext;

double tanh(double x) noexcept;
double tanh_derivative(double x) noexcept;
FCiphertext fhe_tanh(FCiphertext x) noexcept;
FCiphertext fhe_tanh_derivative(FCiphertext x) noexcept;

double sigmoid(double x) noexcept;
double sigmoid_derivative(double x) noexcept;
FCiphertext fhe_sigmoid(FCiphertext x) noexcept;
FCiphertext fhe_sigmoid_derivative(FCiphertext x) noexcept;

double relu(double x) noexcept;
double relu_derivative(double x) noexcept;
FCiphertext fhe_relu(FCiphertext x) noexcept;
FCiphertext fhe_relu_derivative(FCiphertext x) noexcept;

double identity(double x) noexcept;
double identity_derivative(double x) noexcept;
FCiphertext fhe_identity(FCiphertext x) noexcept;
FCiphertext fhe_identity_derivative(FCiphertext x) noexcept;

}; // namespace F

namespace NN {
class LinearLayer {
  private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    const F::Activation activation;
    const F::Activation derivative;
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> deltas;
    const size_t input_size;
    const size_t output_size;

  public:
    LinearLayer(size_t input_size, size_t output_size, F::Activation f);
    std::vector<double> forward(const std::vector<double>& x);
    void _forward(const std::vector<double>& x, std::vector<double>& result);
    void initialize_params() noexcept;
    void compute_deltas(const std::vector<double>& target_deltas,
                        const std::vector<std::vector<double>>& next_weights);
    void update_weights(double learning_rate);

    friend class MLP;
};
class MLP {
  private:
    std::vector<LinearLayer> layers;

  public:
    MLP(const std::vector<size_t>& szs);
    MLP(const std::vector<size_t>& szs, const std::vector<F::Activation>& fs);
    std::vector<double> forward(const std::vector<double>& x);
    void _forward(const std::vector<double>& x, std::vector<double>& result);
    double update(const std::vector<double>& input,
                  const std::vector<double>& target,
                  const double learning_rate);
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>> Y, const int epochs,
               const double learning_rate, bool verbose);
    friend class FHE_MLP;
};
class FHE_MLP {};

} // namespace NN