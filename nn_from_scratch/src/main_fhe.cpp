#include "nn.h"
#include "nn_fhe.h"
#include <cstdio>
#include <ctime>
#include <iostream>

#define INPUT_DIM 4
#define OUTPUT_DIM 2

using namespace lbcrypto;
using namespace bigintdyn;

void save_to_csv(std::vector<std::vector<double>> inputs,
                 std::vector<std::vector<double>> outputs,
                 std::string filename) {
  assert(inputs.size() == outputs.size());
  std::ofstream outputFile(filename);
  for (size_t i = 0; i < inputs.size(); i++) {
    for (int j = 0; j < INPUT_DIM; j++) {
      outputFile << inputs[i][j] << ",";
    }
    for (int j = 0; j < OUTPUT_DIM - 1; j++) {
      outputFile << outputs[i][j] << ",";
    }
    outputFile << outputs[i][OUTPUT_DIM - 1] << std::endl;
  }
  outputFile.close();
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
make_data(size_t N_SAMPLES) {

  const size_t N_ACTIONS = 100;
  std::vector<std::vector<double>> inputs(N_SAMPLES,
                                          std::vector<double>(INPUT_DIM));
  std::vector<std::vector<double>> outputs(N_SAMPLES,
                                           std::vector<double>(OUTPUT_DIM));
  std::vector<std::vector<double>> action_samples(
      N_ACTIONS, std::vector<double>(OUTPUT_DIM));

  for (size_t i = 0; i < N_ACTIONS; i++) {
    for (size_t j = 0; j < OUTPUT_DIM; j++) {
      action_samples[i][j] = F_fhe::random_weight(-1.0, 1.0);
    }
  }
  for (size_t i = 0; i < N_SAMPLES; i++) {
    for (size_t j = 0; j < INPUT_DIM; j++) {
      inputs[i][j] = F_fhe::random_weight(-1.0, 1.0);
    }
    size_t best_j = 0;
    double best_dist = 1000000.0;
    for (size_t j = 0; j < N_ACTIONS; j++) {
      std::vector<double> dist(2);
      dist[0] = inputs[i][0] + action_samples[j][0] - inputs[i][2];
      dist[1] = inputs[i][1] + action_samples[j][1] - inputs[i][3];
      auto norm = std::accumulate(dist.begin(), dist.end(), 0.0,
                                  [](double a, double b) { return a + b * b; });
      if (norm < best_dist) {
        best_dist = norm;
        best_j = j;
      }
    }
    outputs[i] = action_samples[best_j];
  }
  return {inputs, outputs};
}

int main() {
  srand(time(0));
  // begin unencrypted pretraining
  std::vector<size_t> layer_szs = {INPUT_DIM, 3, 3, OUTPUT_DIM};
  std::vector<std::unique_ptr<F::Functional>> fs;
  for (int i = 0; i < (int)layer_szs.size() - 2; i++) {
    fs.emplace_back(std::make_unique<F::Identity>());
  }
  fs.emplace_back(std::make_unique<F::Identity>());

  nn::MLP mlp(layer_szs, fs);

  auto [INPUTS, OUTPUTS] = make_data(10000);

  std::cout << "training...\n";
  for (int epoch = 0; epoch < 10000; epoch++) {
    double net_loss = 0.0;
    for (size_t i = 0; i < INPUTS.size(); i++) {
      net_loss += mlp.update(INPUTS[i], OUTPUTS[i], 0.0001);
    }
    if (epoch % 100 == 0) {
      std::cout << "Epoch : " << epoch << " Loss : " << net_loss << "\n";
    }
  }
  std::cout << "done training\n";
  INPUTS.resize(10);
  // Testing
  std::vector<std::vector<double>> outputs(INPUTS.size(),
                                           std::vector<double>(OUTPUT_DIM));
  for (size_t i = 0; const auto &input : INPUTS) {
    printf("\rInferred: %zu/%zu", i, INPUTS.size());
    outputs[i++] = mlp.forward(input);
  }
  std::cout << "\n";
  save_to_csv(INPUTS, outputs, "unencrypted.csv");

  // done with unencrypted, now lets make an encrypted model and try to run
  // inference
  uint32_t multDepth = 16;
  uint32_t scaleModSize = 50;
  uint32_t batchSize = 1;
  CCParams<CryptoContextCKKSRNS> parameters;
  parameters.SetMultiplicativeDepth(multDepth);
  parameters.SetScalingModSize(scaleModSize);
  parameters.SetBatchSize(batchSize);
  CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->EvalRotateKeyGen(keys.secretKey, {1, -2});
  nn_fhe::MLP fhe_mlp(layer_szs, cc);
  fhe_mlp.set_parameters(mlp);
  for (size_t j = 0; const auto &input : INPUTS) {
    printf("\rInferred: %zu/%zu", j, INPUTS.size());
    std::vector<FunctionalCiphertext> vs(input.size());
    size_t i = 0;
    std::vector<double> inferred(OUTPUT_DIM);
    for (auto in : input) {
      std::vector<double> _in = {in};
      auto plaintext_input = cc->MakeCKKSPackedPlaintext(_in);
      auto encrypted_input = cc->Encrypt(keys.publicKey, plaintext_input);
      vs[i++] = encrypted_input;
    }
    auto encrypted_output = fhe_mlp.forward(vs);
    for (size_t k = 0; auto out : encrypted_output) {
      Plaintext result;
      cc->Decrypt(keys.secretKey, out, &result);
      result->SetLength(batchSize);
      auto vec = result->GetCKKSPackedValue();
      inferred[k++] = vec[0].real();
    }
    outputs[j++] = inferred;
  }
  save_to_csv(INPUTS, outputs, "encrypted.csv");
  return 0;
}