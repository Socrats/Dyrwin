//
// Created by Elias Fernandez on 14/03/2018.
//

#include "../include/CRDPlayer.h"

int CRDPlayer::getAction() {
    return 0;
}

void CRDPlayer::updatePayoff(double curr_payoff) {
    payoff += curr_payoff;
}

int CRDPlayer::getAction(double &public_account) {
    int total_threshold = 120;
    if (public_account < strategy.threshold * total_threshold) {
        return strategy.first;
    } else return strategy.second;
}

void CRDPlayer::reset() {
    payoff = 0;
}

double CRDPlayer::getPayoff() {
    return payoff;
}
