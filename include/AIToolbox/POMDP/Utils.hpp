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
        VList crossSum(const VList & a, const VList & b);

        /**
         * @brief This function prunes all ValueFunctions in the VList that are dominated by others.
         *
         * This function performs simple comparisons between all ValueFunctions in the VList,
         * and is thus much more performant than the prune() function, since that needs to solve
         * multiple linear programming problems. However, this function will not return the truly
         * parsimonious set of ValueFunctions, as its pruning powers are limited.
         *
         * @param pw The list that needs to be pruned.
         */
        void dominationPrune(VList * pw);

        /**
         * @brief This function finds and moves all best ValueFunctions in the simplex corners.
         *
         * What this function does is to find out which ValueFunctions give the highest value in
         * corner beliefs. Since multiple corners may use the same ValueFunction, the number of
         * found ValueFunctions may not be the same as the number of corners. This function
         * moves all found ValueFunctions between the returned iterator and the provided end iterator
         * (excluded).
         *
         * @param S The number of corners of the simplex.
         * @param begin The begin of the search range.
         * @param end The end of the search range. It is NOT included in the search.
         *
         * @return The begin of the range ending with 'end' that contains all found ValueFunctions.
         */
        VList::iterator findBestAtSimplexCorners(size_t S, VList::iterator begin, VList::iterator end);

        /**
         * @brief This function finds the ValueFunction with the highest value for the given belief.
         *
         * @param belief The belief to be used with the ValueFunctions.
         * @param begin The begin of the search range.
         * @param end The range end to be checked. It is NOT included in the search.
         *
         * @return The iterator pointing to the element with the highest dot product with the input belief.
         */
        VList::iterator findBestVector(const Belief & belief, VList::iterator begin, VList::iterator end);

        /**
         * @brief This function finds a witness point where the ValueFunction provided is better than any ValueFunction in the VList.
         *
         * @param v The ValueFunction that may have a witness point.
         * @param best The set of ValueFunctions that may or may not be optimal with respect to v.
         *
         * @return A pair containing true and the found witness point if such a witness exists, and false and empty vector otherwise.
         */
        std::pair<bool, Belief> findWitnessPoint(const MDP::ValueFunction & v, const VList & best);

        /**
         * @brief This function prunes all non useful ValueFunctions from the provided VList.
         *
         * @param w The list that needs to be pruned.
         */
        void prune(VList * w);

    }
}

#endif
