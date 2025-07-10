/*
linear algebra library for HFENN
*/

#include <fhenn/fhenn.h>

namespace FHENN {

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> initialize_openfhe() {
  lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
  parameters.SetMultiplicativeDepth(_CRYPTO_CONTEXT_MULT_DEPTH);
  parameters.SetScalingModSize(_CRYPTO_CONTEXT_SCALE_MOD_SIZE);
  parameters.SetBatchSize(_CRYPTO_CONTEXT_BATCH_SIZE);

  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc =
      lbcrypto::GenCryptoContext(parameters);

  cc->Enable(lbcrypto::PKE);
  cc->Enable(lbcrypto::KEYSWITCH);
  cc->Enable(lbcrypto::LEVELEDSHE);
  return cc;
}

Matrix::Matrix(const std::vector<std::vector<double>> &v)
    : cols(v), n_cols(v.size()) {
  if (n_cols > 0) {
    n_rows = v[0].size();
  } else {
    n_rows = 0;
  }
}

Matrix::~Matrix() {}

std::vector<double> &Matrix::operator[](const int &i) { return cols[i]; }

Matrix Matrix::operator+(Matrix &a) {
  auto b = Matrix{cols};
  if (a.n_cols == n_cols && a.n_rows == n_rows) {
    for (size_t i = 0; i < n_cols; i++) {
      std::transform(cols[i].begin(), cols[i].end(), a[i].begin(), b[i].begin(),
                     std::plus<double>());
    }
  }
  return b;
}

Matrix Matrix::operator-(Matrix &a) {
  auto b = Matrix{cols};
  if (a.n_cols == n_cols && a.n_rows == n_rows) {
    for (size_t i = 0; i < n_cols; i++) {
      std::transform(cols[i].begin(), cols[i].end(), a[i].begin(), b[i].begin(),
                     std::minus<double>());
    }
  }
  return b;
}

// this is where the fun begins
Matrix Matrix::operator*(Matrix &a) {
  std::vector<std::vector<double>> v;
  if (n_cols != a.n_rows) {
    return Matrix({});
  }
  v.reserve(a.n_cols);
  for (size_t i = 0; i < a.n_cols; i++) {
    v.push_back(std::vector<double>(n_rows, 0.0));
  }
  Matrix m(v);
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t k = 0; k < a.n_rows; k++) {
      for (size_t j = 0; j < a.n_cols; j++) {
        m[i][j] += cols[i][k] * a[k][j];
      }
    }
  }
  return m;
}

std::tuple<size_t, size_t> Matrix::shape() { return {n_rows, n_cols}; }

}; // namespace FHENN
