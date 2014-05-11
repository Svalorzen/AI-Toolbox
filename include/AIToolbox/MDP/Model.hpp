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
                 * All rewards are set to 0. The discount parameter is set to 1.
                 *
                 * @param s The number of states of the world.
                 * @param a The number of actions available to the agent.
                 * @param d The discount factor for the MDP.
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
                 * operator[]. In addition, the dimensions of the
                 * containers must match the ones provided as arguments
                 * (for three dimensions: s,a,s).
                 *
                 * This is important, as this constructor DOES NOT perform
                 * any size checks on the external containers.
                 *
                 * Internal values of the containers will be converted to double,
                 * so these convertions must be possible.
                 *
                 * In addition, the transition container must contain a
                 * valid transition function.
                 * \sa transitionCheck()
                 *
                 * \sa copyTable3D()
                 *
                 * The discount parameter must be between 0 and 1 included, otherwise
                 * the constructor will throw an std::invalid_argument.
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
                 * This function will throw a std::invalid_argument if the table provided
                 * does not respect the constraints specified in the mdpCheck() function.
                 *
                 * The container needs to support data access through
                 * operator[]. In addition, the dimensions of the
                 * container must match the ones provided as arguments
                 * (for three dimensions: s,a,s).
                 *
                 * This is important, as this constructor DOES NOT perform
                 * any size checks on the external container.
                 *
                 * Internal values of the container will be converted to double,
                 * so that convertion must be possible.
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
                 * operator[]. In addition, the dimensions of the
                 * containers must match the ones provided as arguments
                 * (for three dimensions: s,a,s).
                 *
                 * This is important, as this constructor DOES NOT perform
                 * any size checks on the external containers.
                 *
                 * Internal values of the container will be converted to double,
                 * so that convertion must be possible.
                 *
                 * @tparam T The external transition container type.
                 * @param t The external transitions container.
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
                 * This function samples the model for simulate experience. The transition
                 * and reward functions are used to produce, from the state action pair
                 * inserted as arguments, a possible new state with respective reward.
                 * The new state is picked from all possible states that the MDP allows
                 * transitioning to, each with probability equal to the same probability
                 * of the transition in the model. After a new state is picked, the reward
                 * is the corresponding reward contained in the reward function.
                 *
                 * @param s The state that needs to be sampled.
                 * @param a The action that needs to be sampled.
                 *
                 * @return A tuple containing a new state and a reward.
                 */
                std::pair<size_t, double> sample(size_t s, size_t a) const;

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

            private:
                size_t S, A;
                double discount_;

                TransitionTable transitions_;
                RewardTable rewards_;

                mutable std::default_random_engine rand_;
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
                    if ( ! isProbability(t[s][a], S) ) throw std::invalid_argument("Input transition table does not contain valid probabilities.");

            copyTable3D(t, transitions_, S, A, S);
        }

        template <typename R>
        void Model::setRewardFunction( const R & r ) {
            copyTable3D(r, rewards_, S, A, S);
        }

    } // MDP
} // AIToolbox

#endif
