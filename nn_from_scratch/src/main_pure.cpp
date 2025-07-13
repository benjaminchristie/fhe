#include "nn_pure.h"
#include <cstdio>
#include <ctime>
#include <iostream>

#define INPUT_DIM 4
#define OUTPUT_DIM 2

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
            action_samples[i][j] = utils::rand_double(-1.0, 1.0);
        }
    }
    for (size_t i = 0; i < N_SAMPLES; i++) {
        for (size_t j = 0; j < INPUT_DIM; j++) {
            inputs[i][j] = utils::rand_double(-1.0, 1.0);
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

void print_vec(const std::vector<double>& v) {
    std::cout << " ( ";
    for (size_t i = 0; i < v.size() - 1; i++) {
        std::cout << v[i] << ", ";
    }
    std::cout << v[v.size() - 1] << " ) ";
}

int main() {
    srand(time(0));
    std::vector<size_t> szs = {INPUT_DIM, 4, 4, 3, OUTPUT_DIM};
    std::vector<F::Activation> fs = {F::relu, F::relu, F::relu, F::identity};
    NN::MLP mlp(szs, fs);
    auto [INPUTS, OUTPUTS] = make_data(1000);
    std::cout << "Beginning training...\n";
    mlp.train(INPUTS, OUTPUTS, 10000, 0.001, true);
    for (size_t i = 0; i < 10; i++) {
        std::cout << "Input:";
        print_vec(INPUTS[i]);
        std::cout << "Predicted:";
        auto v = mlp.forward(INPUTS[i]);
        print_vec(v);
        std::cout << "Output:";
        print_vec(OUTPUTS[i]);
        std::cout << std::endl;
    }
    return 0;
}