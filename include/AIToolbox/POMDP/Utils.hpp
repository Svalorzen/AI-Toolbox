#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/POMDP/Types.hpp>

// This file contains a number of important functions that will most probably be reused
// between different POMDP algorithms, and are thus provided globally.
namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This function outputs the cross summation between two lists of ValueFunctions.
         *
         * @param a The first list.
         * @param b The second list.
         *
         * @return A new list containing all combinations of the elements of a and b summed.
         */
        VList crossSum(size_t S, size_t a, const VList & l1, const VList & l2);
    }
}

#endif
