#include "nn_fhe.h"

namespace F_fhe {

double random_weight() noexcept { return ((double)rand() / RAND_MAX) * 2 - 1; }

double random_weight(const double min, const double max) noexcept {
  return ((double)rand() / RAND_MAX) * (max - min) + min;
}

std::vector<double> vec_dot(std::vector<double> a,
                            std::vector<double> b) noexcept {
  assert(a.size() == b.size());
  size_t sz = a.size();
  std::vector<double> v(sz);
  for (size_t i = 0; i < sz; i++) {
    v[i] = a[i] * b[i];
  }
  return v;
}

std::vector<double> vec_add(std::vector<double> a,
                            std::vector<double> b) noexcept {
  assert(a.size() == b.size());
  size_t sz = a.size();
  std::vector<double> v(sz);
  for (size_t i = 0; i < sz; i++) {
    v[i] = a[i] + b[i];
  }
  return v;
}

std::vector<double> vec_sub(std::vector<double> a,
                            std::vector<double> b) noexcept {
  assert(a.size() == b.size());
  size_t sz = a.size();
  std::vector<double> v(sz);
  for (size_t i = 0; i < sz; i++) {
    v[i] = a[i] - b[i];
  }
  return v;
}

std::vector<double> scalar_dot(double s, std::vector<double> a) noexcept {
  size_t sz = a.size();
  std::vector<double> v(sz);
  std::transform(a.begin(), a.end(), v.begin(),
                 [&](double val) { return s * val; });
  return v;
}

std::vector<double> scalar_div(double s, std::vector<double> a) noexcept {
  return scalar_dot(1.0 / s, a);
}

std::vector<double> scalar_add(double s, std::vector<double> a) noexcept {
  size_t sz = a.size();
  std::vector<double> v(sz);
  std::transform(a.begin(), a.end(), v.begin(),
                 [&](double val) { return s + val; });
  return v;
}

std::vector<double> scalar_sub(double s, std::vector<double> a) noexcept {
  size_t sz = a.size();
  std::vector<double> v(sz);
  std::transform(a.begin(), a.end(), v.begin(),
                 [&](double val) { return s - val; });
  return v;
}

std::vector<double> scalar_sub(std::vector<double> a, double s) noexcept {
  size_t sz = a.size();
  std::vector<double> v(sz);
  std::transform(a.begin(), a.end(), v.begin(),
                 [&](double val) { return val - s; });
  return v;
}

FunctionalCiphertext Functional::forward(FunctionalCiphertext x) noexcept {
  return x;
}
FunctionalCiphertext Functional::derivative(FunctionalCiphertext x) noexcept {
  return x;
}

std::vector<FunctionalCiphertext>
Functional::forward(std::vector<FunctionalCiphertext> x) noexcept {
  std::vector<FunctionalCiphertext> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 [&](FunctionalCiphertext val) { return forward(val); });
  return y;
}

std::vector<FunctionalCiphertext>
Functional::derivative(std::vector<FunctionalCiphertext> x) noexcept {
  std::vector<FunctionalCiphertext> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 [&](FunctionalCiphertext val) { return derivative(val); });
  return y;
}

FunctionalCiphertext Identity::forward(FunctionalCiphertext x) noexcept {
  return x;
}
FunctionalCiphertext Identity::derivative(FunctionalCiphertext x) noexcept {
  auto zero = cc->EvalSub(x, x);
  auto result = cc->EvalAdd(zero, 1.0);
  return result;
}
FunctionalCiphertext ReLU::forward(FunctionalCiphertext x) noexcept {
  // see https://arxiv.org/pdf/2011.05530
  auto x_squared = cc->EvalSquare(x);
  auto result = cc->EvalAdd(x_squared, x);
  return result;
}
FunctionalCiphertext ReLU::derivative(FunctionalCiphertext x) noexcept {
  auto mult = cc->EvalMult(x, 2.0);
  auto result = cc->EvalAdd(mult, 1.0);
  return result;
}
FunctionalCiphertext Sigmoid::forward(FunctionalCiphertext x) noexcept {
  // taylor expansion centered at 0 with O(5) terms
  auto x_2 = cc->EvalSquare(x);
  auto x_3 = cc->EvalMult(x, x_2);
  auto x_5 = cc->EvalMult(x_3, x_2);
  auto t1 = static_cast<double>(0.5);
  auto t2 = cc->EvalMult(x, static_cast<double>(0.25));
  auto t3 = cc->EvalMult(x_3, static_cast<double>(-1.0 / 48.0));
  auto t4 = cc->EvalMult(x_5, static_cast<double>(1.0 / 480.0));
  auto result = cc->EvalAdd(t1, t2);
  result = cc->EvalAdd(t3, result);
  result = cc->EvalAdd(t4, result);
  return result;
}
FunctionalCiphertext Sigmoid::derivative(FunctionalCiphertext x) noexcept {
  // derivative of taylor expansion centered at 0 with O(5) terms
  auto x_2 = cc->EvalSquare(x);
  auto x_4 = cc->EvalSquare(x_2);
  auto t1 = static_cast<double>(0.25);
  auto t2 = cc->EvalMult(x_2, static_cast<double>(-1.0 / (48.0 / 3.0)));
  auto t3 = cc->EvalMult(x_4, static_cast<double>(1.0 / (480.0 / 5.0)));
  auto result = cc->EvalAdd(t1, t2);
  result = cc->EvalAdd(t3, result);
  return result;
}
FunctionalCiphertext Tanh::forward(FunctionalCiphertext x) noexcept {
  // taylor expansion centered at 0 with O(5) terms
  auto x_2 = cc->EvalSquare(x);
  auto x_3 = cc->EvalMult(x, x_2);
  auto x_5 = cc->EvalMult(x_3, x_2);
  auto t1 = x;
  auto t2 = cc->EvalMult(x_3, static_cast<double>(-1.0 / 3.0));
  auto t3 = cc->EvalMult(x_5, static_cast<double>(2.0 / 15.0));
  auto result = cc->EvalAdd(t1, t2);
  result = cc->EvalAdd(result, t3);
  return result;
}
FunctionalCiphertext Tanh::derivative(FunctionalCiphertext x) noexcept {
  // derivative of taylor expansion centered at 0 with O(5) terms
  auto x_2 = cc->EvalSquare(x);
  auto x_4 = cc->EvalSquare(x_2);
  auto t1 = static_cast<double>(1.0);
  auto t2 = cc->EvalMult(x_2, static_cast<double>(-1.0));
  auto t3 = cc->EvalMult(x_4, static_cast<double>(2.0 / 3.0));
  auto result = cc->EvalAdd(t1, t2);
  result = cc->EvalAdd(t3, result);
  return result;
}

} // namespace F_fhe