#ifndef AI_TOOLBOX_MDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_MDP_MODEL_HEADER_FILE

#include <utility>
#include <random>

#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This class represents a Markov Decision Process.
         *
         * A Markov Decision Process (MDP) is a way to model decision making.
         * The idea is that there is an agent situated in a stochastic
         * environment which changes in discrete "timesteps". The agent can
         * influence the way the environment changes via "actions". For each
         * action the agent can perform, the environment will transition from a
         * state "s" to a state "s1" following a certain transition function.
         * The transition function specifies, for each triple SxAxS' the
         * probability that such a transition will happen.
         *
         * In addition, associated with transitions, the agent is able to
         * obtain rewards. Thus, if it does good, the agent will obtain a
         * higher reward than if it performed badly. The reward obtained by the
         * agent is in addition associated with a "discount" factor: at every
         * step, the possible reward that the agent can collect is multiplied
         * by this factor, which is a number between 0 and 1. The discount
         * factor is used to model the fact that often it is preferable to
         * obtain something sooner, rather than later.
         *
         * Since all of this is governed by probabilities, it is possible to
         * solve an MDP model in order to obtain an "optimal policy", which is
         * a way to select an action from a state which will maximize the
         * expected reward that the agent is going to collect during its life.
         * The expected reward is computed as the sum of every reward the agent
         * collects at every timestep, keeping in mind that at every timestep
         * the reward is further and further discounted.
         *
         * Solving an MDP in such a way is called "planning". Planning
         * solutions often include an "horizon", which is the number of
         * timesteps that are included in an episode. They can be finite or
         * infinite. The optimal policy changes with respect to the horizon,
         * since a higher horizon may offer access to reward-gaining
         * opportunities farther in the future.
         *
         * An MDP policy (be it the optimal one or another), is associated with
         * two functions: a ValueFunction and a QFunction. The ValueFunction
         * represents the expected return for the agent from any initial state,
         * given that actions are going to be selected according to the policy.
         * The QFunction is similar: it gives the expected return for a
         * specific state-action pair, given that after the specified action
         * one will act according to the policy.
         *
         * Given that we are usually interested about the optimal policy, there
         * are a couple of properties that are associated with the optimal
         * policies functions.  First, the optimal policy can be derived from
         * the optimal QFunction. The optimal policy simply selects, in a given
         * state "s", the action that maximizes the value of the QFunction.  In
         * the same way, the optimal ValueFunction can be computed from the
         * optimal QFunction by selecting the max with respect to the action.
         *
         * Since so much information can be extracted from the QFunction, lots
         * of methods (mostly in Reinforcement Learning) try to learn it.
         */
        class Model {
            public:
                using TransitionTable   = Table3D;
                using RewardTable       = Table3D;

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor initializes the Model so that all
                 * transitions happen with probability 0 but for transitions
                 * that bring back to the same state, no matter the action.
                 *
                 * All rewards are set to 0. The discount parameter is set to
                 * 1.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param discount The discount factor for the MDP.
                 */
                Model(size_t s, size_t a, double discount = 1.0);

                /**
                 * @brief Basic constructor.
                 *
                 * This constructor takes two arbitrary three dimensional
                 * containers and tries to copy their contents into the
                 * transitions and rewards tables respectively.
                 *
                 * The containers need to support data access through
                 * operator[]. In addition, the dimensions of the containers
                 * must match the ones provided as arguments (for three
                 * dimensions: s,a,s).
                 *
                 * This is important, as this constructor DOES NOT perform any
                 * size checks on the external containers.
                 *
                 * Internal values of the containers will be converted to
                 * double, so these conversions must be possible.
                 *
                 * In addition, the transition container must contain a valid
                 * transition function.  \sa transitionCheck()
                 *
                 * \sa copyTable3D()
                 *
                 * The discount parameter must be between 0 and 1 included,
                 * otherwise the constructor will throw an
                 * std::invalid_argument.
                 *
                 * @tparam T The external transition container type.
                 * @tparam R The external rewards container type.
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param t The external transitions container.
                 * @param r The external rewards container.
                 * @param d The discount factor for the MDP.
                 */
                template <typename T, typename R>
                Model(size_t s, size_t a, const T & t, const R & r, double d = 1.0);

                /**
                 * @brief This function replaces the Model transition function with the one provided.
                 *
                 * This function will throw a std::invalid_argument if the
                 * table provided does not respect the constraints specified in
                 * the mdpCheck() function.
                 *
                 * The container needs to support data access through
                 * operator[]. In addition, the dimensions of the container
                 * must match the ones provided as arguments (for three
                 * dimensions: s,a,s).
                 *
                 * This is important, as this constructor DOES NOT perform any
                 * size checks on the external container.
                 *
                 * Internal values of the container will be converted to
                 * double, so these conversions must be possible.
                 *
                 * @tparam T The external transition container type.
                 * @param t The external transitions container.
                 */
                template <typename T>
                void setTransitionFunction(const T & t);

                /**
                 * @brief This function replaces the Model reward function with the one provided.
                 *
                 * The container needs to support data access through
                 * operator[]. In addition, the dimensions of the containers
                 * must match the ones provided as arguments (for three
                 * dimensions: s,a,s).
                 *
                 * This is important, as this constructor DOES NOT perform any
                 * size checks on the external containers.
                 *
                 * Internal values of the container will be converted to
                 * double, so these conversions must be possible.
                 *
                 * @tparam R The external rewards container type.
                 * @param r The external rewards container.
                 */
                template <typename R>
                void setRewardFunction(const R & r);

                /**
                 * @brief This function sets a new discount factor for the Model.
                 *
                 * @param d The new discount factor for the Model.
                 */
                void setDiscount(double d);

                /**
                 * @brief This function samples the MDP for the specified state action pair.
                 *
                 * This function samples the model for simulated experience.
                 * The transition and reward functions are used to produce,
                 * from the state action pair inserted as arguments, a possible
                 * new state with respective reward.  The new state is picked
                 * from all possible states that the MDP allows transitioning
                 * to, each with probability equal to the same probability of
                 * the transition in the model. After a new state is picked,
                 * the reward is the corresponding reward contained in the
                 * reward function.
                 *
                 * @param s The state that needs to be sampled.
                 * @param a The action that needs to be sampled.
                 *
                 * @return A tuple containing a new state and a reward.
                 */
                std::tuple<size_t, double> sampleSR(size_t s, size_t a) const;

                /**
                 * @brief This function returns the number of states of the world.
                 *
                 * @return The total number of states.
                 */
                size_t getS() const;

                /**
                 * @brief This function returns the number of available actions to the agent.
                 *
                 * @return The total number of actions.
                 */
                size_t getA() const;

                /**
                 * @brief This function returns the currently set discount factor.
                 *
                 * @return The currently set discount factor.
                 */
                double getDiscount() const;

                /**
                 * @brief This function returns the stored transition probability for the specified transition.
                 *
                 * @param s The initial state of the transition.
                 * @param a The action performed in the transition.
                 * @param s1 The final state of the transition.
                 *
                 * @return The probability of the specified transition.
                 */
                double getTransitionProbability(size_t s, size_t a, size_t s1) const;

                /**
                 * @brief This function returns the stored expected reward for the specified transition.
                 *
                 * @param s The initial state of the transition.
                 * @param a The action performed in the transition.
                 * @param s1 The final state of the transition.
                 *
                 * @return The expected reward of the specified transition.
                 */
                double getExpectedReward(size_t s, size_t a, size_t s1) const;

                /**
                 * @brief This function returns the transition table for inspection.
                 *
                 * @return The rewards table.
                 */
                const TransitionTable & getTransitionFunction() const;

                /**
                 * @brief This function returns the rewards table for inspection.
                 *
                 * @return The rewards table.
                 */
                const RewardTable &     getRewardFunction()     const;

                /**
                 * @brief This function returns whether a given state is a terminal.
                 *
                 * @param s The state examined.
                 *
                 * @return True if the input state is a terminal, false otherwise.
                 */
                bool isTerminal(size_t s) const;

            private:
                size_t S, A;
                double discount_;

                TransitionTable transitions_;
                RewardTable rewards_;

                mutable std::default_random_engine rand_;

                friend std::istream& operator>>(std::istream &is, Model &);
        };

        template <typename T, typename R>
        Model::Model(size_t s, size_t a, const T & t, const R & r, double d) : S(s), A(a), transitions_(boost::extents[S][A][S]), rewards_(boost::extents[S][A][S]) {
            setDiscount(d);
            setTransitionFunction(t);
            setRewardFunction(r);
        }

        template <typename T>
        void Model::setTransitionFunction(const T & t) {
            for ( size_t s = 0; s < S; ++s )
                for ( size_t a = 0; a < A; ++a )
                    if ( ! isProbability(S, t[s][a]) ) throw std::invalid_argument("Input transition table does not contain valid probabilities.");

            copyTable3D(t, transitions_, S, A, S);
        }

        template <typename R>
        void Model::setRewardFunction( const R & r ) {
            copyTable3D(r, rewards_, S, A, S);
        }

        std::istream& operator>>(std::istream &is, Model &);
    } // MDP
} // AIToolbox

#endif
