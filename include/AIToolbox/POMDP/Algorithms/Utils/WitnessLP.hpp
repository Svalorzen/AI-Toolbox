#ifndef AI_TOOLBOX_POMDP_WITNESS_LP_HEADER_FILE
#define AI_TOOLBOX_POMDP_WITNESS_LP_HEADER_FILE

#include <optional>

#include <AIToolbox/LP.hpp>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements an easy interface to do Witness discovery through linear programming.
     *
     * This class is meant to help finding witness points by solving the linear
     * programming needed. As such, it contains a linear programming problem where
     * constraints can be set. This class automatically sets the simplex constraint,
     * where a found belief point needs to sum up to one.
     *
     * Optimal constraints can be progressively added as soon as found. When a
     * new constraint needs to be tested to see if a witness is available, the
     * findWitness() function can be called.
     */
    class WitnessLP {
        public:
            /**
             * @brief Basic constructor.
             *
             * This initializes lp_solve structures.
             *
             * @param S The number of states in the world.
             */
            WitnessLP(size_t S);

            /**
             * @brief This function adds a new optimal constraint to the LP, which will not be removed unless the LP is reset.
             *
             * @param v The optimal constraint to add.
             */
            void addOptimalRow(const MDP::Values & v);

            /**
             * @brief This function solves the currently set LP.
             *
             * This function tries to solve the underlying LP, and if
             * successful returns the witness belief point which satisfies
             * the solution.
             *
             * @return If found, the Belief witness to the set problem.
             */
            std::optional<POMDP::Belief> findWitness(const MDP::Values & v);

            /**
             * @brief This function resets the internal LP to only the simplex constraint.
             *
             * This function does not mess with the already allocated memory.
             */
            void reset();

            /**
             * @brief This function reserves space for a certain amount of rows (not counting the simplex) to avoid reallocations.
             *
             * @param rows The max number of constraints for the LP.
             */
            void allocate(size_t rows);

        private:
            size_t S;
            LP lp_;
    };
}

#endif

