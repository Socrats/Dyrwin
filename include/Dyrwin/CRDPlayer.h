/**
 * File: ParameterHandler.h
 * Author: Elias Fernandez
 * Date Created: 14/03/2018
 *
 * CRDPlayer Class. Implements the player encoding defined in M. A. Chakra & A. Traulsen 2012 for the Collective-Risk
 * Dilemma.
 *
 * Each player contains an encoded Strategy which will be optimized through evolution
 *
 */

#ifndef DYRWIN_CRDPLAYER_H
#define DYRWIN_CRDPLAYER_H

#include <random>
#include <boost/functional/hash.hpp>
#include "SeedGenerator.h"


struct Strategy {
    int first;
    int second;
    double threshold;

    Strategy operator=(const Strategy &other) const {
        // Enforces that the reference to the random number generator is not changed
        return *this;
    }

    bool operator==(const Strategy &other) const {
        return (first == other.first
                && second == other.second
                && threshold == other.threshold);
    }

    bool operator!=(const Strategy &other) const {
        return !(first == other.first
                 && second == other.second
                 && threshold == other.threshold);
    }

    Strategy(double mu, double sigma) : _mu(mu), _sigma(sigma) {

        std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
        std::uniform_int_distribution<unsigned int> _uniform_int_dist(0, 2);
        std::normal_distribution<double> _normal_dist(_mu, _sigma);

        // Strategies actions are initialized at random with 0, 1, or 2
        // and the thresholds are uniformly distributed between 0 and 1
        threshold = _uniform_real_dist(_mt);
        first = _uniform_int_dist(_mt);
        second = _uniform_int_dist(_mt);
    };

    Strategy(int first, int second, double threshold, double mu, double sigma) :
            first(first), second(second), threshold(threshold), _mu(mu), _sigma(sigma) {};

    Strategy &operator++() {
        // Mutate strategy - Prefix
        std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
        std::uniform_int_distribution<unsigned int> _uniform_int_dist(0, 2);
        std::normal_distribution<double> _normal_dist(_mu, _sigma);

        // Mutate strategy - Postfix
        if (_uniform_real_dist(_mt) < _mu) {
            threshold += _normal_dist(_mt);
        }
        if (_uniform_real_dist(_mt) < _mu) {
            first = _uniform_int_dist(_mt);
            first = _uniform_int_dist(_mt);
        }
        return *this;
    }

    Strategy operator++(int) {
        std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
        std::uniform_int_distribution<unsigned int> _uniform_int_dist(0, 2);
        std::normal_distribution<double> _normal_dist(_mu, _sigma);

        // Mutate strategy - Postfix
        if (_uniform_real_dist(_mt) < _mu) {
            threshold += _normal_dist(_mt);
        }
        if (_uniform_real_dist(_mt) < _mu) {
            first = _uniform_int_dist(_mt);
            first = _uniform_int_dist(_mt);
        }
        return *this;
    }

    void copy(Strategy &other) {
        first = other.first;
        second = other.second;
        threshold = other.threshold;
    }

private:
    std::mt19937_64 _mt{SeedGenerator::getSeed()};

    // mutation probabilities
    double _mu; // mutation probability
    double _sigma; // standard deviation of the error

};

struct SequentialStrategy {
    /**
     * Implements player described in Maria A. Chakra et al. Plos paper.
     */
    std::vector<Strategy> round_strategies;
    size_t rounds;

    bool operator==(const SequentialStrategy &other) const {
        bool equal = true;
        for (size_t i = 0; i < rounds; i++) {
            if (round_strategies[i] == other.round_strategies[i]) {
                equal = false;
                break;
            }
        }
        return equal;
    }

    SequentialStrategy(double mu, double sigma) : rounds(10) {
        round_strategies.reserve(rounds);
        for (size_t i = 0; i < rounds; i++) {
            round_strategies.push_back(Strategy(mu, sigma));
        }
    };

    SequentialStrategy(double mu, double sigma, size_t rounds) : rounds(rounds) {
        round_strategies.reserve(rounds);
        for (size_t i = 0; i < rounds; i++) {
            round_strategies.push_back(Strategy(mu, sigma));
        }
    };

    SequentialStrategy(std::vector<Strategy> &strategies, size_t rounds) :
            round_strategies(strategies), rounds(rounds) {};

    SequentialStrategy &operator++() {
        // Mutate strategy - Prefix
        for (auto &round_strategy: round_strategies) {
            round_strategy++;
        }
        return *this;
    }

    SequentialStrategy operator++(int) {
        // Mutate strategy - Postfix
        for (auto &round_strategy: round_strategies) {
            round_strategy++;
        }
        return *this;
    }

    void copy(SequentialStrategy &other) {
        for (size_t i = 0; i < rounds; i++) {
            round_strategies[i].copy(other.round_strategies[i]);
        }
    }
};

struct StrategyHasher {
    std::size_t operator()(const Strategy &s) const {
        using boost::hash_value;
        using boost::hash_combine;

        // Start with a hash value of 0    .
        std::size_t seed = 0;

        // Modify 'seed' by XORing and bit-shifting in
        // one member of 'Key' after the other:
        hash_combine(seed, hash_value(s.first));
        hash_combine(seed, hash_value(s.second));
        hash_combine(seed, hash_value(s.threshold));

        // Return the result.
        return seed;
    }
};

struct SequentialStrategyHasher {
    std::size_t operator()(const SequentialStrategy &s) const {
        using boost::hash_value;
        using boost::hash_combine;

        // Start with a hash value of 0    .
        std::size_t seed = 0;

        // Modify 'seed' by XORing and bit-shifting in
        // one member of 'Key' after the other:
        for (size_t i = 0; i < s.rounds; i++) {
            hash_combine(seed, hash_value(s.round_strategies[i].first));
            hash_combine(seed, hash_value(s.round_strategies[i].second));
            hash_combine(seed, hash_value(s.round_strategies[i].threshold));
        }
        hash_combine(seed, hash_value(s.rounds));

        // Return the result.
        return seed;
    }
};

class CRDPlayer {
    /**
     * Implements the player described in Maria A. Chacra et al. Plos paper.
     */
public:
    /**
	    \brief Constructor

	    This constructor generates an ID for this player, and resets its
	    score.
	*/
    CRDPlayer(double mu, double sigma) :
            id(CRDPlayer::GenerateID()),
            payoff(0), strategy(SequentialStrategy(mu, sigma)) {};

    virtual ~CRDPlayer() {};

    /**
	    \brief Copy constructor

	    Initialize a player as an exact copy of a previous player, including
	    the unique ID.

	    \param p Player to be copied
	*/
    CRDPlayer(const CRDPlayer &p) :
            id(p.id),
            payoff(p.payoff), strategy(SequentialStrategy(p.mu, p.sigma)) {}

    int getAction();

    int getAction(double &public_account, size_t rd);

    void updatePayoff(double curr_payoff);

    void reset(double endowment);

    static int GenerateID() {
        static int idCounter = 111;
        return (idCounter++);
    }

    double getPayoff();

    SequentialStrategy strategy;

    double mu, sigma;

    CRDPlayer operator=(const CRDPlayer &other) const {
        // Enforces that the reference to the random number generator is not changed
        return *this;
    }

protected:
    /**
        \brief This player's unique identifier

        This value is an identifier unique to this execution of
        Oyun, used to tell the difference between players in a
        tournament.
    */
    int id;
    /**
    \brief This player's score for this game

    This value accumulates over the course of a \c Match (which
    consists of many repeated games).
    */
    double payoff;

};


#endif //DYRWIN_CRDPLAYER_H
