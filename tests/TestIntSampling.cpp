//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <random>
#include <vector>

int main() {

    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_int_distribution<size_t> int_sampler(0, 49);
    std::vector<size_t> strategies{0, 49, 0, 0, 1};

    auto player1 = int_sampler(generator);
    auto player2 = int_sampler(generator);
    while (player2 == player1) player2 = int_sampler(generator);

    size_t tmp = 0;
    size_t s1 = 0;
    size_t s2 = 0;
    bool unset_p1 = true, unset_p2 = true;

    for (size_t i = 0; i < strategies.size(); ++i) {
        tmp += strategies[i];
        if (tmp > player1 && unset_p1) {
            s1 = i;
            unset_p1 = false;
        }
        if (tmp > player2 && unset_p2) {
            s2 = i;
            unset_p2 = false;
        }
        if (!unset_p1 && !unset_p2) break;
    }
    std::cout << "strategy 1: (" << player1 << ", " << s1 << ")" << ", strategy 2: (" << player2 << ", " << s2 << ")"
              << std::endl;

    return 0;
}
