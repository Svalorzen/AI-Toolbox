#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <cstddef>
#include <iterator>
#include <iostream>
#include <numeric>

#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
    namespace POMDP {
        /**
         * @brief This function creates an empty VEntry.
         *
         * @param S The number of possible states.
         * @param a The action contained in the VEntry.
         * @param O The size of the observations vector.
         *
         * @return A new VEntry.
         */
        VEntry makeVEntry(size_t S, size_t a = 0, size_t O = 0);

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
        template <typename BeliefIterator, typename Iterator>
        Iterator findBestAtBelief(BeliefIterator bbegin, BeliefIterator bend, Iterator begin, Iterator end) {
            auto bestMatch = begin;
            double bestValue = std::inner_product(bbegin, bend, std::begin(std::get<VALUES>(*bestMatch)), 0.0);

            while ( (++begin) < end ) {
                auto & v = std::get<VALUES>(*begin);
                double currValue = std::inner_product(bbegin, bend, std::begin(v), 0.0);
                if ( currValue > bestValue || ( currValue == bestValue && ( v > std::get<VALUES>(*bestMatch) ) ) ) {
                    bestMatch = begin;
                    bestValue = currValue;
                }
            }

            return bestMatch;
        }

        /**
         * @brief This function finds and moves the ValueFunction with the highest value for the given belief at the end of the specified range.
         *
         * This function uses an already existing bound containing previously marked useful
         * ValueFunctions. The order is 'begin'->'bound'->'end', where bound may be equal to end
         * where no previous bound exists. The found ValueFunction is moved between 'bound' and
         * 'end', but only if it was not there previously.
         *
         * @tparam Iterator An iterator, can be const or not, from VList.
         * @param S The number of states for the Belief/Values.
         * @param belief The belief to be used with the ValueFunctions.
         * @param begin The begin of the search range.
         * @param bound The begin of the 'useful' range.
         * @param end The range end to be checked. It is NOT included in the search.
         *
         * @return The iterator pointing to the element with the highest dot product with the input belief.
         */
        template <typename BeliefIterator, typename Iterator>
        Iterator extractBestAtBelief(BeliefIterator bbegin, BeliefIterator bbend, Iterator begin, Iterator bound, Iterator end) {
            // It makes more sense to look from end to start, since the best are there already
            auto bestMatch = findBestAtBelief(bbegin, bbend, std::reverse_iterator<Iterator>(end), std::reverse_iterator<Iterator>(begin));

            // Note that we can do the +1 thing because we know that bestMatch is NOT rend
            if ( (bestMatch+1).base() < bound )
                std::swap(*bestMatch, *(--bound));

            return bound;
        }

        /**
         * @brief Creates a new belief reflecting changes after an action and observation for a particular Model.
         *
         * This function needs to create a new belief since modifying a belief
         * in place is not possible. This is because each cell update for the
         * new belief requires all values from the previous belief.
         *
         * @tparam M The type of the POMDP Model.
         * @param model The model used to update the belief.
         * @param b The old belief.
         * @param a The action taken during the transition.
         * @param o The observation registered.
         */
        template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
        Belief updateBelief(const M & model, const Belief & b, size_t a, size_t o) {
            size_t S = model.getS();
            Belief br(S, 0.0);

            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    sum += model.getTransitionProbability(s,a,s1) * b[s];

                br[s1] = model.getObservationProbability(s1,a,o) * sum;
            }

            normalizeProbability(std::begin(br), std::end(br), std::begin(br));

            return br;
        }
    }
}

#endif
