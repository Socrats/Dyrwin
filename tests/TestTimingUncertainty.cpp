//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <Dyrwin/RL/TimingUncertainty.hpp>


using namespace std;
using namespace EGTTools;

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    int sum = 0;

    EGTTools::TimingUncertainty<std::mt19937> tu(1. / 3);

    std::map<int, int> hist;
    for(int n=0; n<10000; ++n) {
        ++hist[tu.calculateFullEnd(8, gen)];
    }
    for(auto p : hist) {
        std::cout << p.first <<
                  ' ' << std::string(p.second/100, '*') << '\n';
        sum += p.second * p.first;
    }
    auto mean = static_cast<double>(sum) / 10000;
    std::cout << "mean = " << mean << std::endl;
    // Checking with quite a big tolerance (10%)
    assert(mean < 10 + 0.1);
    assert(mean > 10 - 0.1);

    return 0;
}
