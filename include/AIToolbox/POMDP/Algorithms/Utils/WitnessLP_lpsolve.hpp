#ifndef AI_TOOLBOX_POMDP_LP_HEADER_FILE
#define AI_TOOLBOX_POMDP_LP_HEADER_FILE

#include <cstddef>
#include <memory>
#include <vector>

#include <AIToolbox/POMDP/Types.hpp>

#include <lpsolve/lp_types.h>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This class implements easy-to-use facilities to do linear programming.
         *
         * This particular implementation of the class uses lp_solve to do linear
         * programming.
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
        class WitnessLP_lpsolve {
            public:
                /**
                 * @brief Basic constructor.
                 *
                 * This initializes lp_solve structures.
                 *
                 * @param S The number of states in the world.
                 */
                WitnessLP_lpsolve(size_t S);

                /**
                 * @brief This function adds a new optimal constraint to the LP, which will not be removed unless the LP is reset.
                 *
                 * @param v The optimal constraint to add.
                 */
                void addOptimalRow(const std::vector<double> & v);

                /**
                 * @brief This function solves the currently set LP.
                 *
                 * This function tries to solve the underlying LP, and
                 * returns whether a solution has been found. If it is
                 * it also returns the witness belief point which satisfies
                 * the solution.
                 *
                 * @return A pair of whether a solution has been found, and an eventual Belief with the solution.
                 */
                std::tuple<bool, POMDP::Belief> findWitness(const std::vector<double> & v);

                /**
                 * @brief This function resets the LP and reserves space for a certain amount of rows to avoid reallocations.
                 *
                 * @param rows The max number of constraints for the LP.
                 */
                void resetAndAllocate(size_t rows);

            private:
                /**
                 * @brief This function adds a constraint row in the LP.
                 *
                 * This class works as a stack of constraints: one can
                 * only add and remove constraints at the bottom of the
                 * stack. This works out fine for the work this class is
                 * designed to do, which is finding witness points with
                 * respect to a set of already optimal constraints which
                 * will not need to be removed.
                 *
                 * @param v The constraint coefficients.
                 * @param constrType The constraint type.
                 */
                void pushRow(const std::vector<double> & v, int constrType);

                /**
                 * @brief This function removes a single constraint from the LP in a LIFO fashion.
                 */
                void popRow();

                size_t S;
                int cols;
                std::unique_ptr<lprec, void(*)(lprec*)> lp;
                std::unique_ptr<REAL[]> row;
        };
    }
}

#endif
