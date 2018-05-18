//
// Created by Elias Fernandez on 14/03/2018.
//

#include "../include/Dyrwin/CRDPlayer.h"

int CRDPlayer::getAction() {
    return 0;
}

void CRDPlayer::updatePayoff(double curr_payoff) {
    payoff = curr_payoff;
}

int CRDPlayer::getAction(double &public_account, size_t rd, double &threshold) {
    if ((strategy.round_strategies[rd].threshold * threshold) > threshold) {
        payoff -=  strategy.round_strategies[rd].second;
        return  strategy.round_strategies[rd].second;
    } else {
        payoff -=  strategy.round_strategies[rd].first;
        return  strategy.round_strategies[rd].first;
    }
}

void CRDPlayer::reset(double endowment) {
    payoff = endowment;
}

double CRDPlayer::getPayoff() {
    return payoff;
}
