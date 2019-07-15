//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>


using namespace std;

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::geometric_distribution<size_t> geometric(0.1);

    std::map<int, int> hist;
    for(int n=0; n<10000; ++n) {
        ++hist[geometric(gen)];
    }
    for(auto p : hist) {
        std::cout << p.first <<
                  ' ' << std::string(p.second, '*') << '\n';
    }

    return 0;
}
