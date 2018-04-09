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


struct Strategy {
    int first;
    int second;
    double threshold;

    bool operator==(const Strategy &other) const {
        return (first == other.first
                && second == other.second
                && threshold == other.threshold);
    }

    Strategy() : first(0), second(0), threshold(0) {};

    Strategy(int first, int second, double threshold) :
            first(first), second(second), threshold(threshold) {};

    Strategy &operator++() {
        // Mutate strategy - Prefix
        return *this;
    }

    Strategy operator++(int) {
        // Mutate strategy - Postfix
        return *this;
    }

    void copy(Strategy &other) {
        first = other.first;
        second = other.second;
        threshold = other.threshold;
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

class CRDPlayer {
public:
    /**
	    \brief Constructor

	    This constructor generates an ID for this player, and resets its
	    score.
	*/
    CRDPlayer() :
            id(CRDPlayer::GenerateID()),
            payoff(0), strategy(Strategy()) {};

    virtual ~CRDPlayer() {};

    /**
	    \brief Copy constructor

	    Initialize a player as an exact copy of a previous player, including
	    the unique ID.

	    \param p Player to be copied
	*/
    CRDPlayer(const CRDPlayer &p) :
            id(p.id),
            payoff(p.payoff) {}

    int getAction();

    int getAction(double &public_account);

    void updatePayoff(double curr_payoff);

    void reset(double endowment);

    static int GenerateID() {
        static int idCounter = 111;
        return (idCounter++);
    }

    double getPayoff();

    Strategy strategy;

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
