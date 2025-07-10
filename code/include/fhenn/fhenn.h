#pragma once

#include "openfhe.h"
#include <algorithm>
#include <tuple>
#include <vector>

#define _CRYPTO_CONTEXT_SCALE_MOD_SIZE 50
#define _CRYPTO_CONTEXT_BATCH_SIZE 8
#define _CRYPTO_CONTEXT_MULT_DEPTH 1

namespace FHENN {
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> initialize_openfhe();

class Matrix {
private:
  std::vector<std::vector<double>> cols;
  size_t n_cols;
  size_t n_rows;

public:
  Matrix(const std::vector<std::vector<double>> &v);
  ~Matrix();
  Matrix operator+(Matrix &a);
  Matrix operator-(Matrix &a);
  Matrix operator*(Matrix &a);
  std::vector<double> &operator[](const int &i);
  std::tuple<size_t, size_t> shape();
}; // class Matrix

}; // namespace FHENN