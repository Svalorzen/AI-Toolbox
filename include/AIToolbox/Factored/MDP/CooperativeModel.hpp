#ifndef AI_TOOLBOX_FACTORED_COOPERATIVE_MDP_MODEL_HEADER_FILE
#define AI_TOOLBOX_FACTORED_COOPERATIVE_MDP_MODEL_HEADER_FILE

#include <AIToolbox/Factored/MDP/Types.hpp>
#include <AIToolbox/Factored/Utils/FactoredMatrix.hpp>
#include <AIToolbox/Factored/Utils/BayesianNetwork.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class models a cooperative MDP.
     *
     * This class can be used in order to model problems where multiple agents
     * cooperate in order to achieve a common goal. In particular, we model
     * problems where each agent only cares about a specific subset of the
     * state space, which allows to build a coordination graph to store
     * dependencies.
     */
    class CooperativeModel {
        public:
            /**
             * @brief Basic constructor
             *
             * @param s The state space.
             * @param a The action space.
             * @param transitions The transition function.
             * @param rewards The reward function.
             */
            CooperativeModel(State s, Action a, FactoredDDN transitions, Factored2DMatrix rewards);

            /**
             * @brief This function sets a new discount factor for the Model.
             *
             * @param d The new discount factor for the Model.
             */
            void setDiscount(double d);

            /**
             * @brief This function returns the state space of the world.
             *
             * @return The state space.
             */
            const State & getS() const;

            /**
             * @brief This function returns the action space of the MDP.
             *
             * @return The action space.
             */
            const Action & getA() const;

            /**
             * @brief This function returns the currently set discount factor.
             *
             * @return The currently set discount factor.
             */
            double getDiscount() const;

            /**
             * @brief This function returns the transition function of the MDP.
             *
             * @return The transition function of the MDP.
             */
            const FactoredDDN & getTransitionFunction() const;
            const Factored2DMatrix & getRewardFunction() const;
        private:
            State S;
            Action A;
            double discount_;

            FactoredDDN transitions_;
            Factored2DMatrix rewards_;
    };
}

#endif
