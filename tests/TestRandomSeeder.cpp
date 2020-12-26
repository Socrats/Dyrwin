//
// Created by Elias Fernandez on 25/12/2020.
//
#include <random>
#include <iostream>

template<class T = std::mt19937, std::size_t N = T::state_size * sizeof(typename T::result_type)>
auto SeededRandomEngine() -> typename std::enable_if<N == 1, T>::type {
  std::random_device source;
  return T(source());
}

template<class T = std::mt19937, std::size_t N = T::state_size * sizeof(typename T::result_type)>
auto SeededRandomEngine() -> typename std::enable_if<N >= 2, T>::type {
  std::random_device source;
  std::random_device::result_type random_data[(N - 1) / sizeof(source()) + 1];
  std::generate(std::begin(random_data), std::end(random_data), std::ref(source));
  std::seed_seq seeds(std::begin(random_data), std::end(random_data));

  std::cout << (N - 1) / sizeof(source()) + 1 << std::endl;
  std::cout << seeds.size() << std::endl;

  return T(seeds);
}

int main() {

  auto seeder = SeededRandomEngine<std::mt19937_64>();
  std::cout << seeder() << std::endl;
  std::cout << seeder() << std::endl;
  auto seeder_2 = SeededRandomEngine<std::mt19937_64, 1>();
  std::cout << seeder_2() << std::endl;
  return 0;
}