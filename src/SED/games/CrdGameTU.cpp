//
// Created by Elias Fernandez on 2019-06-27.
//

#include <Dyrwin/SED/games/CrdGameTU.hpp>

EGTTools::SED::CRD::CrdGameTU::CrdGameTU(size_t endowment, size_t threshold, size_t min_rounds, size_t group_size,
                                         double risk,
                                         EGTTools::TimingUncertainty<std::mt19937_64> &tu)
        : endowment_(endowment),
          threshold_(threshold), min_rounds_(min_rounds), group_size_(group_size), risk_(risk), tu_(tu) {
    nb_strategies_ = nb_strategies();
    // number of possible group combinations
    nb_states_ = EGTTools::starsBars(group_size_, nb_strategies_);
    // number of possible group combinations without the focal player
    nb_states_player_ = EGTTools::starsBars(group_size - 1, nb_strategies_);
    payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_states_);

    // initialise random distribution
    real_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);

    // Initialise payoff matrix
    calculate_payoffs();
}

void
EGTTools::SED::CRD::CrdGameTU::play(const EGTTools::SED::StrategyCounts &group_composition,
                                    PayoffVector &game_payoffs) {
    size_t prev_donation = 0, current_donation = 0;
    size_t public_account = 0;
    size_t player_aspiration = (group_size_ - 1) * 2;
    size_t game_rounds = tu_.calculateFullEnd(min_rounds_, generator_);
    VectorXui actions = VectorXui::Zero(nb_strategies_);
    // Initialize payoffs
    for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
        if (group_composition[j] > 0) {
            game_payoffs[j] = endowment_;
        } else {
            game_payoffs[j] = 0;
        }
    }

    for (size_t i = 0; i < game_rounds; ++i) {
        for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
            if (group_composition[j] > 0) {
                actions(j) = get_action(j, prev_donation - actions(j), player_aspiration, i);
                if (game_payoffs[j] - actions(j) >= 0) {
                    game_payoffs[j] -= actions(j);
                    current_donation += group_composition[j] * actions(j);
                }
            }
        }
        public_account += current_donation;
        prev_donation = current_donation;
        current_donation = 0;
        if (public_account >= threshold_) break;
    }

    if (public_account < threshold_)
        for (auto &type: game_payoffs) type *= risk_;
}

size_t
EGTTools::SED::CRD::CrdGameTU::get_action(const size_t &player_type, const size_t &prev_donation,
                                          const size_t &threshold,
                                          const size_t &current_round) {
    switch (player_type) {
        case 0:
            return EGTTools::SED::CRD::cooperator(prev_donation, threshold, current_round);
        case 1:
            return EGTTools::SED::CRD::defector(prev_donation, threshold, current_round);
        case 2:
            return EGTTools::SED::CRD::altruist(prev_donation, threshold, current_round);
        case 3:
            return EGTTools::SED::CRD::reciprocal(prev_donation, threshold, current_round);
        case 4:
            return EGTTools::SED::CRD::compensator(prev_donation, threshold, current_round);
        default:
            assert(false);
            throw std::invalid_argument("invalid player type: " + std::to_string(player_type));
    }
}

std::string EGTTools::SED::CRD::CrdGameTU::toString() const {
    return "Collective-risk dilemma game with Timing uncertainty.\n"
           "It only plays the game with the strategies described in EGTtools::SED::CRD::behaviors";
}

std::string EGTTools::SED::CRD::CrdGameTU::type() const {
    return "CrdGameTU";
}

size_t EGTTools::SED::CRD::CrdGameTU::nb_strategies() const {
    return EGTTools::SED::CRD::nb_strategies;
}

const EGTTools::SED::GroupPayoffs &EGTTools::SED::CRD::CrdGameTU::calculate_payoffs() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);

    // For every possible group composition run the game and store the payoff of each strategy
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update group composition from current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // Since the number of rounds of the game is stochastic
        // we repeat the game a 1000 times to obtain a good estimation
        for (size_t z = 0; z < 1000; ++z) {
            // play game and update game_payoffs
            play(group_composition, game_payoffs);

            // Fill payoff table
            for (size_t j = 0; j < nb_strategies_; ++j) payoffs_(j, i) += game_payoffs[j];
        }
        payoffs_.col(i) /= 1000;
    }

    return payoffs_;
}

double EGTTools::SED::CRD::CrdGameTU::calculate_fitness(const size_t &player_type, const size_t &pop_size,
                                                        const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double fitness = 0.0, payoff;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // If the focal player is not in the group, then the payoff should be zero
        if (sample_counts[player_type] > 0) {

            // First update sample_counts with new group composition
            payoff = payoffs_(player_type, EGTTools::SED::calculate_state(group_size_, sample_counts));
            sample_counts[player_type] -= 1;

            // Calculate probability of encountering a the current group
            auto prob = EGTTools::multivariateHypergeometricPDF(pop_size - 1, nb_strategies_, group_size_ - 1,
                                                                sample_counts,
                                                                strategies);
            sample_counts[player_type] += 1;

            fitness += payoff * prob;
        }
    }

    return fitness;
}

void EGTTools::SED::CRD::CrdGameTU::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << payoffs_ << std::endl;
        file << "group_size = " << group_size_ << std::endl;
        file << "timing_uncertainty = false" << std::endl;
        file << "min_rounds = " << min_rounds_ << std::endl;
        file << "p = " << tu_.probability() << std::endl;
        file << "risk = " << risk_ << std::endl;
        file << "endowment = " << endowment_ << std::endl;
        file << "threshold = " << threshold_ << std::endl;
        file.close();
    }
}

const EGTTools::SED::GroupPayoffs &EGTTools::SED::CRD::CrdGameTU::payoffs() const {
    return payoffs_;
}

double
EGTTools::SED::CRD::CrdGameTU::payoff(size_t strategy, const EGTTools::SED::StrategyCounts &group_composition) const {
    return payoffs_(strategy, EGTTools::SED::calculate_state(group_size_, group_composition));
}
