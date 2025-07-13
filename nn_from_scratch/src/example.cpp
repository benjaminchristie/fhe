#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "openfhe.h"
#include <chrono>

using namespace std;
using namespace lbcrypto;
using namespace bigintdyn;

typedef Ciphertext<DCRTPolyImpl<mubintvec<ubint<unsigned long>>>>
    FunctionalCiphertext;

typedef PublicKey<DCRTPolyImpl<mubintvec<ubint<unsigned long>>>> PublicKeyType;

double sigmoid(double x) {
    return 0.5 + x / 4.0 - (x * x * x) / 48.0 + (x * x * x * x * x) / 480.0;
}

double sigmoid_derivative(double x) {
    //   return x * (1.0 - x); // Assumes x is already sigmoid(x)
    return 1.0 / 4.0 - (x * x) / (48.0 / 3.0) + (x * x * x * x) / (480.0 / 5.0);
}

FunctionalCiphertext fhe_sigmoid(CryptoContext<DCRTPoly> cc,
                                 FunctionalCiphertext x) {
    auto x_2 = cc->EvalMult(x, x);
    auto x_3 = cc->EvalMult(x_2, x);
    auto x_5 = cc->EvalMult(x_2, x_3);
    auto t1 = static_cast<double>(0.5);
    auto t2 = cc->EvalMult(x, 0.25);
    auto t3 = cc->EvalMult(x_3, -1.0 / 48.0);
    auto t4 = cc->EvalMult(x_5, 1.0 / 480.0);
    auto result = cc->EvalAdd(t1, t2);
    result = cc->EvalAdd(t3, result);
    result = cc->EvalAdd(t4, result);
    return result;
}

FunctionalCiphertext fhe_sigmoid_derivative(CryptoContext<DCRTPoly> cc,
                                            FunctionalCiphertext x) {
    // x = sigmoid(cc, x);
    // return x * (1.0 - x); // Assumes x is already sigmoid(x)
    auto x_2 = cc->EvalMult(x, x);
    auto x_4 = cc->EvalMult(x_2, x_2);
    auto t1 = static_cast<double>(0.25);
    auto t2 = cc->EvalMult(x_2, -1.0 / (48.0 / 3.0));
    auto t3 = cc->EvalMult(x_4, 1.0 / (480.0 / 5.0));
    auto result = cc->EvalAdd(t1, t2);
    result = cc->EvalAdd(t3, result);
    return result;
}

double randWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Range [-1, 1]
}

class Layer {
  public:
    int input_size, output_size;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> outputs;
    vector<double> inputs;
    vector<double> deltas;

    Layer(int in_size, int out_size)
        : input_size(in_size), output_size(out_size) {
        weights.resize(out_size, vector<double>(in_size));
        biases.resize(out_size);
        outputs.resize(out_size);
        deltas.resize(out_size);
        inputs.resize(in_size);
        // Random initialization
        for (int i = 0; i < out_size; ++i) {
            biases[i] = randWeight();
            for (int j = 0; j < in_size; ++j) {
                weights[i][j] = randWeight();
            }
        }
    }

    vector<double> forward(const vector<double>& input) {
        inputs = input;
        for (int i = 0; i < output_size; ++i) {
            double sum = biases[i];
            for (int j = 0; j < input_size; ++j) {
                sum += weights[i][j] * input[j];
            }
            outputs[i] = sigmoid(sum);
        }
        return outputs;
    }

    void computeDeltas(const vector<double>& target_or_next_deltas,
                       const vector<vector<double>>& next_weights = {}) {
        for (int i = 0; i < output_size; ++i) {
            if (next_weights.empty()) {
                // Output layer
                deltas[i] = (outputs[i] - target_or_next_deltas[i]) *
                            sigmoid_derivative(outputs[i]);
            } else {
                // Hidden layer
                double error = 0.0;
                for (size_t j = 0; j < target_or_next_deltas.size(); ++j) {
                    error += target_or_next_deltas[j] * next_weights[j][i];
                }
                deltas[i] = error * sigmoid_derivative(outputs[i]);
            }
        }
    }

    void updateWeights(double learning_rate) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] -= learning_rate * deltas[i] * inputs[j];
            }
            biases[i] -= learning_rate * deltas[i];
        }
    }
};

class NeuralNetwork {
  public:
    vector<Layer> layers;

    NeuralNetwork(const vector<int>& layer_sizes) {
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            layers.emplace_back(layer_sizes[i - 1], layer_sizes[i]);
        }
    }

    vector<double> predict(const vector<double>& input) {
        vector<double> out = input;
        for (auto& layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    void train(const vector<vector<double>>& X, const vector<vector<double>>& y,
               int epochs, double lr) {
        // for (int epoch = 0; epoch < epochs; ++epoch) {
        //   double total_loss = 0.0;
        for (size_t i = 0; i < X.size(); ++i) {
            vector<double> out = predict(X[i]);

            // // Compute loss
            // for (size_t j = 0; j < out.size(); ++j) {
            //   total_loss += 0.5 * pow(out[j] - y[i][j], 2);
            // }

            // Backpropagation
            layers.back().computeDeltas(y[i]); // Output layer

            for (int l = layers.size() - 2; l >= 0; --l) {
                layers[l].computeDeltas(layers[l + 1].deltas,
                                        layers[l + 1].weights);
            }

            for (auto& layer : layers) {
                layer.updateWeights(lr);
            }
        }
    }
    // }
};

class FHELayer {
  public:
    int input_size, output_size;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<FunctionalCiphertext> outputs;
    vector<FunctionalCiphertext> inputs;
    vector<double> deltas;
    CryptoContext<DCRTPoly> cc;

    FHELayer(int in_size, int out_size, CryptoContext<DCRTPoly> cc)
        : input_size(in_size), output_size(out_size), cc(cc) {
        weights.resize(out_size, vector<double>(in_size));
        biases.resize(out_size);
        outputs.resize(out_size);
        deltas.resize(out_size);
        inputs.resize(in_size);
        // Random initialization
        for (int i = 0; i < out_size; ++i) {
            biases[i] = randWeight();
            for (int j = 0; j < in_size; ++j) {
                weights[i][j] = randWeight();
            }
        }
    }

    vector<FunctionalCiphertext>
    forward(const vector<FunctionalCiphertext>& input,
            const PublicKeyType publicKey) {
        inputs = input;
        for (int i = 0; i < output_size; ++i) {
            std::vector<double> zero = {biases[i]};
            Plaintext plaintext = cc->MakeCKKSPackedPlaintext(zero);
            auto sum = cc->Encrypt(publicKey, plaintext);
            for (int j = 0; j < input_size; ++j) {
                auto curr = cc->EvalMult(input[j], weights[i][j]);
                sum = cc->EvalAdd(curr, sum);
            }
            outputs[i] = fhe_sigmoid(cc, sum);
        }
        return outputs;
    }
};

class FHENeuralNetwork {
  public:
    vector<FHELayer> layers;

    FHENeuralNetwork(const vector<int>& layer_sizes,
                     CryptoContext<DCRTPoly> cc) {
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            layers.emplace_back(layer_sizes[i - 1], layer_sizes[i], cc);
        }
    }

    vector<FunctionalCiphertext> predict(vector<FunctionalCiphertext> input,
                                         const PublicKeyType publicKey) {
        vector<FunctionalCiphertext> out = input;
        for (auto& layer : layers) {
            out = layer.forward(out, publicKey);
        }
        return out;
    }
};

#define INPUT_DIM 4
#define OUTPUT_DIM 2

double random_weight(const double min, const double max) noexcept {
    return ((double)rand() / RAND_MAX) * (max - min) + min;
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
            action_samples[i][j] = random_weight(-1.0, 1.0);
        }
    }
    for (size_t i = 0; i < N_SAMPLES; i++) {
        for (size_t j = 0; j < INPUT_DIM; j++) {
            inputs[i][j] = random_weight(-1.0, 1.0);
        }
        size_t best_j = 0;
        double best_dist = 1000000.0;
        for (size_t j = 0; j < N_ACTIONS; j++) {
            std::vector<double> dist(2);
            dist[0] = inputs[i][0] + action_samples[j][0] - inputs[i][2];
            dist[1] = inputs[i][1] + action_samples[j][1] - inputs[i][3];
            auto norm =
                std::accumulate(dist.begin(), dist.end(), 0.0,
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

    // Example: 2 inputs → 2 hidden → 1 output
    std::vector<int> layer_szs = {INPUT_DIM, 3, 3, OUTPUT_DIM};
    NeuralNetwork nn(layer_szs);

    auto [X, Y] = make_data(1000);

    // std::chrono::steady_clock::time_point begin =
    //     std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < 10000; epoch++) {
        nn.train(X, Y, 10000, 0.001);

        if (epoch % 1000 == 0)
            cout << "Epoch " << epoch << "\n";
    }
    // std::chrono::steady_clock::time_point end =
    //     std::chrono::steady_clock::now();

    // cout << "Training took : "
    //      << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //      begin)
    //             .count()
    //      << "μs\n";

    // X.resize(10);
    // begin = std::chrono::steady_clock::now();
    // for (const auto& input : X) {
    //     vector<double> out = nn.predict(input);
    //     cout << "Input: (";
    //     for (const auto& in : input) {
    //         cout << in << ", ";
    //     }
    //     cout << ") => Output: (";
    //     for (const auto& ou : out) {
    //         cout << ou << ", ";
    //     }
    //     cout << ")\n";
    // }
    // end = std::chrono::steady_clock::now();

    // cout << "Standard inference took : "
    //      << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //      begin)
    //             .count()
    //      << "μs\n";

    // uint32_t multDepth = 10;
    // uint32_t scaleModSize = 50;
    // uint32_t batchSize = 1;

    // CCParams<CryptoContextCKKSRNS> parameters;
    // parameters.SetMultiplicativeDepth(multDepth);
    // parameters.SetScalingModSize(scaleModSize);
    // parameters.SetBatchSize(batchSize);
    // CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    // cc->Enable(PKE);
    // cc->Enable(KEYSWITCH);
    // cc->Enable(LEVELEDSHE);
    // std::cout << "\nCKKS scheme is using ring dimension "
    //           << cc->GetRingDimension() << std::endl;
    // auto keys = cc->KeyGen();
    // cc->EvalMultKeyGen(keys.secretKey);
    // cc->EvalRotateKeyGen(keys.secretKey, {1, -2});
    // FHENeuralNetwork fhenn = FHENeuralNetwork(layer_szs, cc);
    // for (size_t i = 0; i < fhenn.layers.size(); i++) {
    //     fhenn.layers[i].weights = nn.layers[i].weights;
    //     fhenn.layers[i].biases = nn.layers[i].biases;
    // }
    // std::cout << "beginning encrypted inference\n";
    // begin = std::chrono::steady_clock::now();
    // for (const auto& input : X) {
    //     vector<FunctionalCiphertext> vs(input.size());
    //     size_t i = 0;
    //     for (auto in : input) {
    //         std::vector<double> _in = {in};
    //         auto plaintext_input = cc->MakeCKKSPackedPlaintext(_in);
    //         auto encrypted_input = cc->Encrypt(keys.publicKey,
    //         plaintext_input); vs[i++] = encrypted_input;
    //     }
    //     auto encrypted_output = fhenn.predict(vs, keys.publicKey);
    //     cout << "Input: (";
    //     for (const auto& in : input) {
    //         cout << in << ", ";
    //     }
    //     cout << ") => Output: (";
    //     for (auto out : encrypted_output) {
    //         Plaintext result;
    //         cc->Decrypt(keys.secretKey, out, &result);
    //         result->SetLength(batchSize);
    //         auto vec = result->GetCKKSPackedValue();
    //         cout << vec[0].real() << ", ";
    //     }
    //     cout << ")\n";
    // }

    // end = std::chrono::steady_clock::now();

    // cout << "\nFHE inference took : "
    //      << std::chrono::duration_cast<std::chrono::microseconds>(end -
    //      begin)
    //             .count()
    //      << "μs\n";

    return 0;
}
