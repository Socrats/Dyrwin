#include <utility>

//
// Created by Elias Fernandez on 2019-05-10.
//

#ifndef DYRWIN_RL_CRDSIM_HPP
#define DYRWIN_RL_CRDSIM_HPP

#include <random>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/CRDConditional.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/RL/Data.hpp>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::RL {
    class CRDSim {
    public:
        /**
         * @brief Implements a simulator for the Collective Risk dilemma.
         *
         * The main objective of this class is to provide a container for the main
         * simulations that need to be performed in the context of the Collective-risk Dilemma
         * with RL agents.
         *
         * @param nb_episodes number of episodes to be run for each group
         * @param nb_games number of games per episode
         * @param nb_rounds number of rounds per game
         * @param nb_actions number of actions that each player can take
         * @param group_size size of a group
         * @param agent_type algorithm that the agent uses to learn
         * @param risk probability of cataclysm
         * @param args extra arguments for the particular agent_type
         */
        CRDSim(size_t nb_episodes, size_t nb_games,
               size_t nb_rounds,
               size_t nb_actions,
               size_t group_size,
               double risk,
               double endowment,
               double threshold,
               const ActionSpace &available_actions,
               const std::string &agent_type,
               const std::vector<double> &args = {});

        /**
             * @brief Runs a simulation of the CRD for only 1 group.
             *
             * This method runs a simulation of the CRD for @param nb_episodes.
             * Each episode consists of @param nb_games. The policy of the agents
             * is only updated once an episode finishes.
             *
             * @param nb_episodes number of episodes during which the agents will learn
             * @param nb_games number of games per episode
             * @return EGTTools::Matrix2D containing the group achievement and average contribution
             *         for each episode.
             */
        Matrix2D run(size_t nb_episodes, size_t nb_games);

        /**
             * @brief Runs a simulation of the CRD for multiple independent groups.
             *
             * This method runs a simulation of the CRD for @param nb_episodes.
             * Each episode consists of @param nb_games. The policy of the agents
             * is only updated once an episode finishes. The simulation is performed
             * for @param nb_groups independent groups, and the results of each episode
             * are averaged over groups.
             *
             * @param nb_episodes number of episodes during which the agents will learn
             * @param nb_games number of games per episode
             * @param nb_groups number of independent groups
             * @return EGTTools::Matrix2D containing the group achievement and average contribution
             *         for each episode.
             */
        Matrix2D
        run(size_t nb_episodes, size_t nb_games, size_t nb_groups, double risk, const std::vector<double> &args = {});

        /**
             * @brief runs the Collective risk dilemma with independent groups of agents.
             *
             * Trains a population of size @param nb_groups * @param group_size of RL agents of
             * type @param agent_type. The agents are subdivided into @param nb_groups groups
             * and during training time they only interact with members of their group.
             *
             * Each agent's strategy is only updated after each episode.
             *
             * Each group is independently run in parallel. Once the simulation finishes, the
             * learning data is transfered to a CRDData container. The population of agents
             * is also inserted in the container. Finally this method returns the data container.
             *
             * @param nb_groups : number of independent groups
             * @param group_size : size of the group
             * @param nb_episodes : number of episodes
             * @param nb_games : number of games per episode
             * @param risk : probability that agents lose their endowment if the target isn't met
             * @param transient : transient time to be discarded on the computation of the average success
             * @param agent_type : learning algorithm used by the agent
             * @param args : vector of arguments for the creation of the agent
             * @return a data container with the results of the simulation
             */
        DataTypes::CRDDataIslands
        run(size_t nb_groups, size_t group_size, size_t nb_episodes, size_t nb_games, double risk,
            size_t transient, const std::string &agent_type, const std::vector<double> &args = {});

        /**
             * @brief Runs simulation with a well mixed population.
             *
             * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
             * are selected randomly from the population to form a group of size _group_size and play a game.
             * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
             *
             * The @param args is a vector that should contain the arguments specific of the agent type of
             * the population.
             *
             * @param nb_generations : number of generations per simulation
             * @param nb_games : number of games per generation
             * @param nb_groups : will define the population size (Z = nb_groups * _group_size)
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @return an Eigen 2D matrix with the average group achievement and average donations at each generation.
             */
        Matrix2D runWellMixed(size_t nb_generations, size_t nb_games, size_t nb_groups, size_t group_size, double risk,
                              const std::vector<double> &args = {});

        /**
             * @brief Runs simulation with a well mixed population and returns a data container with the population.
             *
             * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
             * are selected randomly from the population to form a group of size _group_size and play a game.
             * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
             *
             * The @param args is a vector that should contain the arguments specific of the agent type of
             * the population.
             *
             * @param pop_size : size of the population
             * @param group_size : group size
             * @param nb_generations : number of generations per simulation
             * @param nb_games : number of games per generation
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param agent_type : string indicating which agent implementation to use
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @return a data container that includes the group achievement, the average donations and the population
             */
        DataTypes::CRDData
        runWellMixed(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                     double risk, const std::string &agent_type,
                     const std::vector<double> &args = {});

        /**
             * @brief Runs several simulations with a well mixed population.
             *
             * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
             * are selected randomly from the population to form a group of size _group_size and play a game.
             * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
             *
             * The @param args is a vector that should contain the arguments specific of the agent type of
             * the population.
             *
             * The simulation is repeated for @param nb_runs with independent populations.
             *
             * @param nb_runs : number of independent simulations
             * @param pop_size : size of the population
             * @param group_size : size of the group
             * @param nb_generations : number of generations per simulation
             * @param nb_games : number of games per generation
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param agent_type : string indicating whcoh agent implementation to use
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @return Eigen 2D matrix with the average group achievement and avg. donations across independent runs.
             */
        Matrix2D runWellMixed(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                              size_t nb_games, double risk, const std::string &agent_type,
                              const std::vector<double> &args = {});

        /**
             * @brief Runs a simulation with timing uncertainty
             *
             * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
             * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
             * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
             *
             * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
             * the average of the geometric distribution that defines the final round is @param avg_rounds.
             *
             * @param nb_episodes : nb_episodes number of episodes during which the agents will learn
             * @param nb_games : number of games per episode
             * @param min_rounds : minimum number of rounds
             * @param mean_rounds : average number of rounds
             * @param max_rounds : maximum number of rounds (used so that the rounds of the game is
             *                     never bigger than the number of states of the agent)
             * @param p : probability of the game ending after each round (starting from min_rounds)
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @param crd_type : type of reinforcement
             * @return an Eigen 2D matrix with the average group achievement and average donations at each episode.
             */
        Matrix2D
        runTimingUncertainty(size_t nb_episodes, size_t nb_games, size_t min_rounds, size_t mean_rounds,
                             size_t max_rounds, double p,
                             double risk,
                             const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

        /**
             * @brief trains a well-mixed population in the CRD with Timing uncertainty
             *
             * In the simulations performed here, agents of a population of size Z = @param pop_size
             * are selected randomly from the population to form a group of size @param group_size and play a game.
             * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
             *
             * The @param args is a vector that should contain the arguments specific of the @param agent_type of
             * the population.
             *
             * The results are transfered to a data container and returned.
             *
             * @param pop_size : size of the population
             * @param group_size : group size
             * @param nb_generations : total number of generations
             * @param nb_games : number of games per generation
             * @param risk : risk of losing the remaining endowment if the target isn't reached
             * @param agent_type : algorithm type of the agent
             * @param args : arguments for the agent
             * @return a data container
             */
        DataTypes::CRDData
        runWellMixedTU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                       double risk, size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                       const std::string &agent_type,
                       const std::vector<double> &args = {});

        Matrix2D
        runWellMixedTU(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                       double risk, size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                       const std::string &agent_type,
                       const std::vector<double> &args = {});

        /**
             * @brief Runs a simulation with threshold uncertainty
             *
             * This simulations run the Collective Risk Game with threshold uncertainty.
             * Here the number of rounds of the game is uncertain. The threshold of the game will be between
             * @param min_threshold and a maximum of @param max_threshold. The threshold will take a value
             * from this range with uniform probability.
             *
             * @param nb_episodes : nb_episodes number of episodes during which the agents will learn
             * @param nb_games : number of games per episode
             * @param min_threshold : minimum threshold
             * @param max_threshold : maximum threshold
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @param crd_type : type of crd
             * @return an Eigen 2D matrix with the average group achievement and average donations at each episode.
             */
        Matrix2D
        runThresholdUncertainty(size_t nb_episodes, size_t nb_games, size_t min_threshold, size_t max_threshold,
                                double risk,
                                const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

        /**
             * @brief Runs a simulation with conditional agents.
             *
             * This method can only be run on actions of the type [0, 1, 2....], i.e., sequential from [0, nb_actions -1].
             *
             * @param nb_episodes : number of times the policy of the agent will be updated based on the propensity matrix.
             * @param nb_games : number of games per episode
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @param crd_type : type of crd game (milinski or xico)
             * @return an Eigen 2D matrix with the average group achievement and average donations at each generation.
             */
        Matrix2D runConditional(size_t nb_episodes, size_t nb_games, const std::vector<double> &args = {},
                                const std::string &crd_type = "milinski");

        DataTypes::CRDData
        runConditional(size_t group_size, size_t nb_episodes, size_t nb_games, double risk,
                       const std::string &agent_type, const std::vector<double> &args = {});

        /**
             * @brief Runs a simulation with timing uncertainty with conditional agents.
             *
             * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
             * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
             * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
             *
             * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
             * the average of the geometric distribution that defines the final round is @param avg_rounds.
             *
             * @param nb_episodes : nb_episodes number of episodes during which the agents will learn
             * @param nb_games : number of games per episode
             * @param min_rounds : minimum number of rounds
             * @param mean_rounds : average number of rounds
             * @param max_rounds : maximum number of rounds (used so that the rounds of the game is
             *                     never bigger than the number of states of the agent)
             * @param p : probability of the game ending after each round (starting from min_rounds)
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @param crd_type : type of reinforcement
             * @return an Eigen 2D matrix with the average group achievement and average donations at each episode.
             */
        Matrix2D
        runConditionalTimingUncertainty(size_t nb_episodes, size_t nb_games, size_t min_rounds, size_t mean_rounds,
                                        size_t max_rounds, double p,
                                        double risk,
                                        const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

        /**
             * @brief Runs several independent simulations with conditional agents.
             *
             * This method can only be run on actions of the type [0, 1, 2....], i.e., sequential from [0, nb_actions -1].
             * A total of @param nb_groups independent simulations will be run. The simulation will return the average
             * group achievement and donations across simulations.
             *
             * @param nb_episodes : number of times the policy of the agent will be updated based on the propensity matrix.
             * @param nb_games : number of games per episode
             * @param nb_groups : number of independent simulations
             * @param risk : probability of loosing all endowment if the target isn't reached
             * @param args : vector of arguments to instantiate the agent_type of the population
             * @param crd_type : type of crd game (milinski or xico)
             * @return Eigene 2D matrix with the average group achievement and average donations across independent runs.
             */
        Matrix2D
        runConditional(size_t nb_episodes, size_t nb_games, size_t nb_groups, double risk,
                       const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

        /**
             * @brief resets the population stored in the class.
             */
        void resetPopulation();

        /**
             * @brief This method reinforces agents only when the target is reached
             * @tparam G : Game class
             * @param pool : crd group with agents
             * @param success : if the group reached the target
             * @param risk : probability of losing all endowment if the target isn't reached
             * @param pop : population container
             * @param game : game object
             */
        template<class G = CRDGame<PopContainer>>
        void reinforceOnlyPositive(double &pool, size_t &success, double &risk, PopContainer &pop,
                                   G &game);

        /**
             * @brief This method reinforces agents only when the target is reached (for Timing uncertainty games)
             * @tparam G : Game class
             * @param pool : crd group with agents
             * @param success : if the group reached the target
             * @param risk : probability of losing all endowment if the target isn't reached
             * @param pop : population container
             * @param final_round : number of rounds of the last played game.
             * @param game : game object
             */
        template<class G = CRDGame<PopContainer>>
        void reinforceOnlyPositive(double &pool, size_t &success, double &risk, PopContainer &pop, size_t &final_round,
                                   G &game);

        /**
             * @brief This method reinforces agents proportionally to the obtained payoff.
             * @tparam G : Game class
             * @param pool : crd group with agents
             * @param success : if the group reached the target
             * @param risk : probability of losing all endowment if the target isn't reached
             * @param pop : population container
             * @param game : game object
             */
        template<class G = CRDGame<PopContainer>>
        void reinforceAll(double &pool, size_t &success, double &risk, PopContainer &pop,
                          G &game);

        /**
             * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
             * @tparam G : Game class
             * @param pool : crd group with agents
             * @param success : if the group reached the target
             * @param risk : probability of losing all endowment if the target isn't reached
             * @param pop : population container
             * @param final_round : number of rounds of the last played game.
             * @param game : game object
             */
        template<class G = CRDGame<PopContainer>>
        void reinforceAll(double &pool, size_t &success, double &risk, PopContainer &pop, size_t &final_round,
                          G &game);

        /**
             * @brief This method reinforces following Xico's version of the CRD payoffs
             * @tparam G : Game class
             * @param pool : crd group with agents
             * @param success :: if the group reached the target
             * @param risk : probability of losing all endowment if the target isn't reached
             * @param pop : population container
             * @param game : game object
             */
        template<class G = CRDGame<PopContainer>>
        void reinforceXico(double &pool, size_t &success, double &risk, PopContainer &pop,
                           G &game);

        /**
             * @brief This method reinforces following Xico's version of the CRD payoffs (for Timing uncertainty games).
             * @tparam G : Game class
             * @param pool : crd group with agents
             * @param success :: if the group reached the target
             * @param risk : probability of losing all endowment if the target isn't reached
             * @param pop : population container
             * @param final_round : number of rounds of the last played game.
             * @param game : game object
             */
        template<class G = CRDGame<PopContainer>>
        void reinforceXico(double &pool, size_t &success, double &risk, PopContainer &pop, size_t &final_round,
                           G &game);

        /**
             * @brief sets the game type to Milinski or Xico
             *
             * If the game type is milinski, the minimum payoff is 0.
             * If it is xico, the minimum payoff is -c*b, i.e., if
             * the group doesn't reach the target, the payoff is
             * - the amount contributed by the player.
             *
             * @param crd_type "milinski" or "xico".
             */
        void setGameType(const std::string &crd_type = "milinski");

        // Getters

        size_t nb_games() const;

        size_t nb_episodes() const;

        size_t nb_rounds() const;

        size_t nb_actions() const;

        double endowment() const;

        double risk() const;

        double threshold() const;

        const ActionSpace &available_actions() const;

        const std::string &agent_type() const;

        // Setters

        void set_nb_games(size_t nb_games);

        void set_nb_episodes(size_t nb_episodes);

        void set_nb_rounds(size_t nb_rounds);

        void set_nb_actions(size_t nb_actions);

        void set_risk(double risk);

        void set_threshold(double threshold);

        void set_available_actions(const ActionSpace &available_actions);

        void set_agent_type(const std::string &agent_type);

        CRDGame<PopContainer> Game;
        PopContainer population;

    private:
        size_t _nb_episodes, _nb_games, _nb_rounds, _nb_actions, _group_size;
        double _risk, _endowment, _threshold;
        ActionSpace _available_actions;
        std::string _agent_type;

        std::uniform_real_distribution<double> _real_rand;

        void (EGTTools::RL::CRDSim::*_reinforce)(double &, size_t &, double &, PopContainer &,
                                                 CRDGame<PopContainer> &) = nullptr;

        // Random generators
        std::mt19937_64 _generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        template<class G>
        void
        reinforceAll(size_t &pool, size_t &success, size_t &threshold, double &risk, PopContainer &pop,
                     size_t &final_round,
                     G &game, std::mt19937_64 &generator);
    };

} // namespace EGTTools::RL

#endif //DYRWIN_RL_CRDSIM_HPP
