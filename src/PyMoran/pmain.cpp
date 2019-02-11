//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "../../include/Dyrwin/SeedGenerator.h"

float fermifunc(float beta, float a, float b);

using namespace std;

int main(int argc, char *argv[]) {

    // First initialize population and global parameters
    unsigned int i, j, w, ref, p1, p2;
    unsigned int pop_size = 50;
    unsigned int nb_betas = 7;
    float freq = 0.5;
    float freq1, freq2;
    vector<unsigned int> population(pop_size);
    float mu = 1e-3;
    float payoffs[4] = {1, 4, 0, 3};
    float fitness1, fitness2 = 0;
    float beta = 1e-4;
    unsigned int generations = 10000;
    unsigned int runs = 100;
    int gradient = 0;
    auto nb_coop = (unsigned int) floor((double) (freq * pop_size));

    // Random generators
    std::mt19937_64 _mt{SeedGenerator::getInstance().getSeed()};
    // Uniform int distribution
    std::uniform_int_distribution<unsigned int> dist(0, pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    clock_t tStart = clock();

    // Betas loop
    for (w = 0; w < nb_betas; w++) {
        cout << "beta: " << beta << endl;

        // Run loop
        for (j = 0; j < runs; j++) {

            ref = nb_coop;

            for (i = 0; i < ref; i++) {
                population[i] = 0;
            }
            for (i = ref; i < pop_size; i++) {
                population[i] = 1;
            }

            // Then we shuffle it randomly
            shuffle(population.begin(), population.end(), _mt);

            // Now run a Moran Process loop
            for (i = 0; i < generations; i++) {
                // Randomly select 2 individuals from the population
                p1 = dist(_mt);
                p2 = dist(_mt);
                while (p2 == p1) p2 = dist(_mt);

                // Calculate frequencies
                freq1 = (ref - population[p1]) / (float) (pop_size - 1);
                freq2 = (ref - population[p2]) / (float) (pop_size - 1);

                // Calculate payoffs
                fitness1 = payoffs[(2 * population[p1]) + 1] * freq1 +
                           payoffs[(2 * population[p1]) + 0] * (1 - freq1);
                fitness2 = payoffs[(2 * population[p2]) + 1] * freq2 +
                           payoffs[(2 * population[p2]) + 0] * (1 - freq2);

                // Select according to fermi function
                if (_uniform_real_dist(_mt) < fermifunc(beta, fitness1, fitness2)) {
                    if (_uniform_real_dist(_mt) < mu) {
                        if (_uniform_real_dist(_mt) < 0.5) {
                            gradient = - population[p1];
                            population[p1] = 0;
                        } else {
                            gradient = 1 - population[p1];
                            population[p1] = 1;
                        }
                    } else {
                        gradient = population[p2] - population[p1];
                        population[p1] = population[p2];
                    }
                } else {
                    if (_uniform_real_dist(_mt) < mu) {
                        if (_uniform_real_dist(_mt) < 0.5) {
                            gradient = - population[p2];
                            population[p1] = 0;
                        } else {
                            gradient = 1 - population[p2];
                            population[p1] = 1;
                        }
                    } else {
                        gradient = population[p1] - population[p2];
                        population[p2] = population[p1];
                    }
                }
                ref += gradient;
                if ((ref == pop_size) || (ref == 0)) break;
            }
            cout << "[" << j << "] Coop freq.: " << ref / (float) pop_size << endl;
        }
        // Update beta
        beta = beta * 10;
    }

    printf("\nExecution time: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}


float fermifunc(float beta, float a, float b) {
    return 1 / (1 + exp(beta * (a - b)));
}
