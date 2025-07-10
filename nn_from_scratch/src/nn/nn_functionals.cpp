#include "nn.h"

namespace F {

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

Functional::Functional() {}

double Functional::forward(double x) noexcept {
  // assert(0 == 1);
  return x;
}
double Functional::derivative(double x) noexcept { return x; }

std::vector<double> Functional::forward(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 [&](double val) { return forward(val); });
  return y;
}

std::vector<double> Functional::derivative(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 [&](double val) { return derivative(val); });
  return y;
}

double Identity::forward(double x) noexcept { return x; }
double Identity::derivative(double x) noexcept { return 1.0; }
double ReLU::forward(double x) noexcept { return x > 0.0 ? x : 0.0; }
double ReLU::derivative(double x) noexcept { return x > 0.0 ? 1.0 : 0.0; }
double Sigmoid::forward(double x) noexcept {
  return 1.0 / (1.0 + std::exp(-x));
}
double Sigmoid::derivative(double x) noexcept {
  auto y = forward(x);
  return y * (1.0 - y);
}
double Tanh::forward(double x) noexcept { return std::tanh(x); }
double Tanh::derivative(double x) noexcept {
  auto y = forward(x);
  return 1.0 - y * y;
}

} // namespace F