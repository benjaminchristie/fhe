#include "nn.h"
#include <iostream>

int main(int argc, char **argv) {

  std::vector<std::unique_ptr<F::Functional>> fs;
  fs.emplace_back(std::make_unique<F::Tanh>());
  fs.emplace_back(std::make_unique<F::Identity>()); // = {
  //       std::make_unique<F::ReLU>(), std::make_unique<F::Identity>()};
  std::cout << fs[0]->forward(1.0) << "\n";
  std::cout << fs[0]->forward(-1.0) << "\n";
  std::cout << fs[1]->forward(1.0) << "\n";
  std::cout << fs[1]->forward(-1.0) << "\n";
  nn::MLP mlp({4, 2, 1}, fs);

  std::vector<std::vector<double>> INPUTS = {
      {1.0, 0.0, 1.0, 0.0},
      {1.0, 0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0, 1.0},
  };
  std::vector<std::vector<double>> OUTPUTS = {
      {1.0},
      {0.0},
      {0.0},
  };

  for (int epoch = 0; epoch < 10000; epoch++) {
    for (size_t i = 0; i < INPUTS.size(); i++) {
      mlp.backward(INPUTS[i], OUTPUTS[i], 0.01);
    }
    if (epoch % 100 == 0) {
      std::cout << "epoch: " << epoch
                << " training: " << mlp.forward(INPUTS[0])[0] << "\n";
    }
  }
  // Testing
  for (const auto &input : INPUTS) {
    auto out = mlp.forward(input);
    std::cout << "Input: ";
    for (double x : input)
      std::cout << x << " ";
    std::cout << " -> Output: ";
    for (double o : out)
      std::cout << o << " ";
    std::cout << std::endl;
  }
  return 0;
}