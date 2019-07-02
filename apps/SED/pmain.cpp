//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <Dyrwin/SED/PDImitation.h>
#include <Dyrwin/SED/StochDynamics.h>
#include <Dyrwin/SED/TraulsenMoran.h>
#include <Dyrwin/SED/MoranProcess.hpp>
#include <Dyrwin/SED/MLS.hpp>
#include <Dyrwin/CommandLineParsing.h>

using namespace std;
using namespace EGTTools;

int main(int argc, char *argv[]) {

    // First initialize population and global parameters
    unsigned int pop_size = 100, nb_betas = 7;
    size_t group_size, nb_groups, runs_grad;
    double split_prob;
    float mu = 1e-3;
    float coop_freq = 0.5;
    Eigen::Matrix2d payoff_matrix;
    payoff_matrix << 1, 4,
            0, 3;
    std::vector<float> betas(nb_betas);
    Vector strategy_freq(2);
    strategy_freq << 0.01, 0.99;
    float beta = 1e-4;
    unsigned int generations = 1000;
    unsigned int runs = 100;
    bool test;
    Options options;

    options.push_back(makeDefaultedOption("generations,g", &generations, "set the number of generations", 1000u));
    options.push_back(makeDefaultedOption("popSize,Z", &pop_size, "set the size of the population", 100u));
    options.push_back(makeDefaultedOption("mu,u", &mu, "set mutation rate", 1e-3f));
    options.push_back(makeDefaultedOption("beta,b", &beta, "set intensity of selection", 1e-4f));
    options.push_back(makeDefaultedOption("runs,R", &runs, "set the number of runs", 100u));
    options.push_back(makeDefaultedOption<size_t>("runsGrad,i", &runs_grad, "set the number of runs", 100));
    options.push_back(makeDefaultedOption("test,t", &test, "test StochDynamics", false));
    options.push_back(makeDefaultedOption("splitProb,q", &split_prob, "split probability", 0.0));
    options.push_back(makeDefaultedOption<size_t>("groupSize,n", &group_size, "split probability", 50));
    options.push_back(makeDefaultedOption<size_t>("nbGroups,m", &nb_groups, "split probability", 1));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    PDImitation pd(generations, pop_size, beta, mu, coop_freq, payoff_matrix);
    TraulsenMoran ts(generations, group_size, nb_groups, beta, mu, coop_freq, split_prob, payoff_matrix);
    MoranProcess mp(generations, 2, pop_size, beta, strategy_freq, payoff_matrix);
    SED::MLS<> multi_sel(10000000, 2, group_size, nb_groups, 0.1, strategy_freq, payoff_matrix);

    clock_t tStart = clock();
    betas[0] = beta;
    for (size_t i = 1; i < nb_betas; i++) {
        betas[i] = betas[i - 1] * 10;
    }

    std::vector<float> result = pd.evolve(betas, runs);
    std::vector<double> result2 = ts.evolve(std::vector<double>(betas.begin(), betas.end()), runs);
    double fix_prob = 0;
    EGTTools::Vector gradient;
    try {
        fix_prob = multi_sel.fixationProbability(1, 0, 10000, split_prob, 0.1);
        multi_sel.set_group_size(10);
        multi_sel.set_nb_groups(10);
        fix_prob = multi_sel.fixationProbability(1, 0, 10000, 0.01, 0., 0.1, 0.025, 0.);
        multi_sel.set_group_size(50);
        multi_sel.set_nb_groups(1);
        multi_sel.fixationProbability(1, (EGTTools::VectorXui(2) << 25, 25).finished(), 10000, split_prob, 0.1);
        gradient = multi_sel.gradientOfSelection(1, 0, runs_grad, 0.1, split_prob);
    } catch (const std::invalid_argument& ia) {
        cerr << "\033[1;31m[EXCEPTION] Invalid argument: " << ia.what() << "\033[0m" << endl;
        return -1;
    }
    Vector result3 = mp.evolve(runs, 0.1);

    cout << "===========" << endl;
    cout << "PDImitation" << endl;
    cout << "===========" << endl;
    for (size_t i = 0; i < nb_betas; i++) {
        cout << "[beta " << betas[i] << "] freq_coop: " << result[i] << endl;
    }

    cout << "===========" << endl;
    cout << "TraulsenMoran" << endl;
    cout << "===========" << endl;
    for (size_t i = 0; i < nb_betas; i++) {
        cout << "[beta " << betas[i] << "] freq_coop: " << result2[i] << endl;
    }

    cout << "===========" << endl;
    cout << "MoranProcess" << endl;
    cout << "===========" << endl;
    cout << "[beta " << 0.1 << "] freq_coop: " << result3 << endl;

    cout << "===========" << endl;
    cout << "    MLS    " << endl;
    cout << "===========" << endl;
    cout << "fixation probability = " << fix_prob << endl;
//    cout << "gradient of selection = " << endl;
//    cout.precision(5);
//    for (size_t i = 0; i < (group_size * nb_groups) + 1; i++) {
//        cout << "[k = " << i << "] gradient: " << gradient(i) << endl;
//    }

    if (test) {
        Eigen::Matrix2d payoffs;
        payoffs << 1, 4,
                0, 3;
        StochDynamics sdy(pop_size, 2, payoffs);
        cout << "Trans prob: " << std::get<1>(sdy.calculate_transition_fixations(1.))(0, 0) << endl;
    }

    printf("\nExecution time: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}
