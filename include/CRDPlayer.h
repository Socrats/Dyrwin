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

#include <boost/functional/hash.hpp>
#include <boost/random.hpp>


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

    Strategy(double mu, double sigma, boost::mt19937& mt) : _mu(mu), _sigma(sigma), _mt(mt) {
        // Strategies actions are initialized at random with 0, 1, or 2
        // and the thresholds are uniformly distributed between 0 and 1
        threshold = _rng_real();
        first = _rng_discrete();
        second = _rng_discrete();
    };

    Strategy(int first, int second, double threshold, double mu, double sigma, boost::mt19937& mt) :
            first(first), second(second), threshold(threshold), _mu(mu), _sigma(sigma), _mt(mt) {};

    Strategy &operator++() {
        // Mutate strategy - Prefix
        if (_rng_real() < _mu) {
            threshold += _rng_normal();
        }
        if (_rng_real() < _mu) {
            first = _rng_discrete();
            first = _rng_discrete();
        }
        return *this;
    }

    Strategy operator++(int) {
        // Mutate strategy - Postfix
        if (_rng_real() < _mu) {
            threshold += _rng_normal();
        }
        if (_rng_real() < _mu) {
            first = _rng_discrete();
            first = _rng_discrete();
        }
        return *this;
    }

    void copy(Strategy &other) {
        first = other.first;
        second = other.second;
        threshold = other.threshold;
    }
private:
    boost::mt19937& _mt;
    boost::random::uniform_real_distribution<> _uniform_real = boost::random::uniform_real_distribution<>(0, 1);
    boost::random::uniform_int_distribution<> _uniform_int = boost::random::uniform_int_distribution<>(0, 2);
    boost::random::normal_distribution<> _normal = boost::random::normal_distribution<>(0.0, _sigma);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<> > _rng_real =
            boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<> >(_mt, _uniform_real);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_int_distribution<> > _rng_discrete =
            boost::variate_generator<boost::mt19937 &, boost::random::uniform_int_distribution<> >(_mt, _uniform_int);
    boost::variate_generator<boost::mt19937 &, boost::random::normal_distribution<> > _rng_normal =
            boost::variate_generator<boost::mt19937 &, boost::random::normal_distribution<> >(_mt, _normal);

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
        for (size_t i=0; i < rounds; i++) {
            if (round_strategies[i] == other.round_strategies[i]) {
                equal = false;
                break;
            }
        }
        return equal;
    }

    SequentialStrategy(double mu, double sigma, boost::mt19937& mt) : rounds(10) {
        round_strategies.reserve(rounds);
        for (size_t i=0; i < rounds; i++) {
            round_strategies.push_back(Strategy(mu, sigma, mt));
        }
    };

    SequentialStrategy(double mu, double sigma, size_t rounds, boost::mt19937& mt) : rounds(rounds) {
        round_strategies.reserve(rounds);
        for (size_t i=0; i < rounds; i++) {
            round_strategies.push_back(Strategy(mu, sigma, mt));
        }
    };

    SequentialStrategy(std::vector<Strategy> &strategies, size_t rounds) :
           round_strategies(strategies), rounds(rounds)  {};

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
        for (size_t i=0; i < s.rounds; i++) {
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
    CRDPlayer(double mu, double sigma, boost::mt19937& mt) :
            id(CRDPlayer::GenerateID()),
            payoff(0), mt(mt), strategy(SequentialStrategy(mu, sigma, mt)) {};

    virtual ~CRDPlayer() {};

    /**
	    \brief Copy constructor

	    Initialize a player as an exact copy of a previous player, including
	    the unique ID.

	    \param p Player to be copied
	*/
    CRDPlayer(const CRDPlayer &p) :
            id(p.id),
            payoff(p.payoff), mt(p.mt), strategy(SequentialStrategy(p.mu, p.sigma, p.mt)) {}

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
    boost::mt19937& mt;

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
