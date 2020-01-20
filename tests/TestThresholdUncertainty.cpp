//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <iomanip>
#include <cassert>
#include <map>
#include <random>

using namespace std;

int main() {

  std::random_device rd;
  std::mt19937 gen(rd());
  size_t sum = 0;
  size_t threshold = 60;
  size_t delta = 20;
  // Define the distribution for the threshold
  std::uniform_int_distribution<size_t> t_dist(threshold - delta / 2, threshold + delta / 2);

  std::map<int, double> hist;
  for (int n = 0; n < 10000; ++n) {
    ++hist[static_cast<double>(t_dist(gen))];
  }
  for (auto p : hist) {
    std::cout << p.first <<
              ' ' << std::string(p.second / 100, '*') << '\n';
    sum += p.second * p.first;
  }
  auto mean = static_cast<double>(sum) / 10000;
  std::cout << "mean = " << mean << std::endl;
  // Checking with quite a big tolerance (10%)
  assert(mean < threshold + 0.3);
  assert(mean > threshold - 0.3);

  return 0;
}
