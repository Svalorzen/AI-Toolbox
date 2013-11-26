#ifndef AI_TOOLBOX_MDP_HEADER_FILE
#define AI_TOOLBOX_MDP_HEADER_FILE

#include <tuple>
#include <random>

#include <AIToolbox/Types.hpp>

namespace AIToolbox {
    /**
     * @brief This class represents a Markov Decision Process.
     *
     */
    class MDP {
        public:
            using TransitionTable   = Table3D;
            using RewardTable       = Table3D;

            /**
             * @brief This function checks whether the supplied table is a correct transition table.
             * 
             * This function verifies basic probability conditions on the
             * supplied container. The sum of all transitions from a 
             * state action pair to all states must be 1.
             *
             * The container needs to support data access through 
             * operator[]. In addition, the dimensions of the
             * container must match the ones provided as arguments
             * (for three dimensions: s,s,a).
             * 
             * This is important, as this function DOES NOT perform
             * any size checks on the external containers.
             *
             * This function is provided so that it is easy to plug
             * this library into existing code-bases.
             *
             * @tparam T The external transition container type.
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param t The external transitions container. 
             *
             * @return True if the container statisfies probability constraints,
             *         and false otherwise.
             */
            template <typename T>
            static bool mdpCheck(size_t s, size_t a, T t);

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
             * (for three dimensions: s,s,a).
             * 
             * This is important, as this constructor DOES NOT perform
             * any size checks on the external containers.
             * 
             * In addition, the transition container must respect
             * the constraint described in the mdpCheck() function.
             *
             * @tparam T The external transition container type.
             * @tparam R The external rewards container type.
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             * @param t The external transitions container. 
             * @param r The external rewards container. 
             */
            template <typename T, typename R>
            MDP(size_t s, size_t a, T t, R r);

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
            std::tuple<size_t, double>  sample(size_t s, size_t a) const;

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
        protected:
            /**
             * @brief Constructor for derived classes.
             * 
             * This constructor is provided as a basic constructor which
             * does not initialize the values contained in the transitions
             * and rewards tables. This is so that derived classes can
             * implement their own initializations without having to pass
             * two container functions into the main constructor.
             *
             * @param s The number of states of the world.
             * @param a The number of actions available to the agent.
             */
            MDP(size_t s, size_t a);

            size_t S, A;

            TransitionTable transitions_;
            RewardTable rewards_;

            mutable std::default_random_engine rand_;
            mutable std::uniform_real_distribution<double> sampleDistribution_;
    };
}

#endif
