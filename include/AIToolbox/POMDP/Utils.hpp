#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <stddef.h>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        VEntry makeVEntry(size_t S, size_t a = 0, size_t O = 0);

        /// THIS IS A TEMPORARY FUNCTION UNTIL WE SWITCH TO UBLAS
        inline double dotProd(size_t S, const MDP::Values & a, const MDP::Values & b) {
            double result = 0.0;

            for ( size_t i = 0; i < S; ++i )
                result += a[i] * b[i];

            return result;
        }

        /**
         * @brief This function returns an iterator pointing to the best value for the specified belief.
         *
         * Ideally I would like to SFINAE that the iterator type is from VList, but at the moment
         * it would take too much time. Just remember that!
         *
         * @tparam Iterator An iterator, can be const or not, from VList.
         * @param S The number of states for the Belief/Values.
         * @param belief The belief to test against.
         * @param begin The start of the range to look in.
         * @param end The end of the range to look in (excluded).
         *
         * @return An iterator pointing to the best choice in range.
         */
        template <typename Iterator>
        Iterator findBestAtBelief(size_t S, const Belief & belief, Iterator begin, Iterator end) {
            auto bestMatch = begin;
            double bestValue = dotProd(S, belief, std::get<VALUES>(*bestMatch));

            while ( (++begin) < end ) {
                double currValue = dotProd(S, belief, std::get<VALUES>(*begin));
                if ( currValue > bestValue || ( currValue == bestValue && ( std::get<VALUES>(*begin) > std::get<VALUES>(*bestMatch) ) ) ) {
                    bestMatch = begin;
                    bestValue = currValue;
                }
            }

            return bestMatch;
        }
    }
}

#endif
