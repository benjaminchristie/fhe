#include "nn.h"

namespace F {

inline double identity(double x) noexcept { return x; }

inline std::vector<double> identity(std::vector<double> x) noexcept {
  return x;
}

inline double identity_derivative(double x) noexcept { return 1.0; }

inline std::vector<double> identity_derivative(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&identity_derivative));
  return y;
}

inline double ReLU(double x) noexcept { return x > 0.0 ? x : 0.0; }

inline std::vector<double> ReLU(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&ReLU));
  return y;
}

inline double ReLU_derivative(double x) noexcept { return x > 0.0 ? 1 : 0.0; }

inline std::vector<double> ReLU_derivative(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&ReLU_derivative));
  return y;
}

inline double sigmoid(double x) noexcept { return 1 / (1 + std::exp(-x)); }

inline std::vector<double> sigmoid(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&sigmoid));
  return y;
}

inline double sigmoid_derivative(double x) noexcept {
  auto y = sigmoid(x);
  return y * (1.0 - y);
}

inline std::vector<double> sigmoid_derivative(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&sigmoid_derivative));
  return y;
}

inline double tanh(double x) noexcept { return std::tanh(x); }

inline std::vector<double> tanh(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&tanh));
  return y;
}

inline double tanh_derivative(double x) noexcept {
  auto y = tanh(x);
  return 1.0 - y * y;
}

inline std::vector<double> tanh_derivative(std::vector<double> x) noexcept {
  std::vector<double> y(x.size());
  std::transform(x.begin(), x.end(), y.begin(),
                 static_cast<double (*)(double)>(&tanh_derivative));
  return y;
}

inline double random_weight() noexcept {
  return ((double)rand() / RAND_MAX) * 2 - 1;
}

inline double random_weight(const double min, const double max) noexcept {
  return ((double)rand() / RAND_MAX) * (max - min) + min;
}

} // namespace F