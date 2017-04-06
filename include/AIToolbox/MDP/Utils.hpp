#ifndef AI_TOOLBOX_MDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_MDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox {
    namespace MDP {
        /**
         * @brief This function creates and zeroes a QFunction.
         *
         * This function exists mostly to avoid remembering how to initialize
         * Eigen matrices, and does nothing special.
         *
         * @param S The state space of the QFunction.
         * @param A The action space of the QFunction.
         *
         * @return A newly built QFunction.
         */
        QFunction makeQFunction(size_t S, size_t A);

        /**
         * @brief This function creates and zeroes a ValueFunction.
         *
         * This function exists mostly to avoid remembering how to initialize
         * Eigen vectors, and does nothing special.
         *
         * @param S The state space of the ValueFunction.
         *
         * @return A newly build ValueFunction.
         */
        ValueFunction makeValueFunction(size_t S);

        /**
         * @brief This function converts a QFunction into the equivalent optimal ValueFunction.
         *
         * The ValueFunction will contain, for each state, the best action and
         * corresponding value as extracted from the input QFunction.
         *
         * @param q The QFunction to convert.
         *
         * @return The equivalent optimal ValueFunction.
         */
        ValueFunction bellmanOperator(const QFunction & q);

        /**
         * @brief This function converts a QFunction into the equivalent optimal ValueFunction.
         *
         * This function is the same as bellmanOperator(), but performs its
         * operations inline. The input ValueFunction MUST already be sized
         * appropriately for the input QFunction.
         *
         * NOTE: This function DOES NOT perform any checks whatsoever on both
         * the validity of the input pointer and on the size of the input
         * ValueFunction. It assumes everything is already correct.
         *
         * @param q The QFunction to convert.
         * @param v A pre-allocated ValueFunction to populate with the optimal values.
         */
        void bellmanOperatorInline(const QFunction & q, ValueFunction * v);
    }
}

#endif
