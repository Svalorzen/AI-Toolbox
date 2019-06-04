#ifndef AI_TOOLBOX_FACTORED_MDP_TIGER_ANTELOPE
#define AI_TOOLBOX_FACTORED_MDP_TIGER_ANTELOPE

#include <AIToolbox/Factored/MDP/CooperativeModel.hpp>

#include <AIToolbox/MDP/Environments/Utils/GridWorld.hpp>

namespace AIToolbox::Factored::MDP {
    /**
     * @brief This class represents a 2-agent tiger antelope environment.
     *
     * The two tigers move in a torus grid which always has the antelope at its
     * center. Their goal is to capture it; this can be done when both tigers
     * are adjacent to the antelope, and *only one of them* moves onto it.
     *
     * The antelope movement is simulated by shifting the whole world around,
     * so that the antelope is always in the "center" of the state-space. This
     * is done to reduce the size of the state-space from a 3d vector to a 2d
     * vector.
     *
     * Each tiger can move in one of the 4 cardinal directions, or stay still.
     */
    class TigerAntelope {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param width The width of the torus.
             * @param height The height of the torus.
             */
            TigerAntelope(unsigned width, unsigned height);

            /**
             * @brief This function allows to sample a new state and rewards.
             *
             * @param s The state to start with.
             * @param a The action to perform.
             *
             * @return A new state and rewards.
             */
            std::tuple<State, Rewards> sampleSRs(const State & s, const Action & a) const;

            /**
             * @brief This function returns whether a state is terminal.
             *
             * Note that this function's return value is not defined for
             * invalid states (for example a state with both tigers in the same
             * place).
             *
             * @param s The state to check.
             *
             * @return Whether the state is terminal.
             */
            bool isTerminalState(const State & s) const;

            /**
             * @brief This function returns the state space of the model.
             */
            State getS() const;

            /**
             * @brief This function returns the action space of the model.
             */
            Action getA() const;

            /**
             * @brief This function returns the discount factor of the model.
             */
            double getDiscount() const;

            /**
             * @brief This function returns the state where the antelope is located.
             */
            size_t getAntelopeState() const;

            /**
             * @brief This function returns a reference to the internal GridWorld.
             */
            const AIToolbox::MDP::GridWorld & getGrid() const;

            /**
             * @brief This function returns a graphical representation of a State.
             *
             * @param s
             *
             * @return 
             */
            std::string printState(const State & s) const;

        private:
            AIToolbox::MDP::GridWorld grid_;
            size_t antelopePosition_;

            mutable RandomEngine rand_;
    };
}

#endif
