#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <cstddef>
#include <iterator>
#include <numeric>

#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/Utils.hpp>
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
         * @brief This function returns a weak measure of distance between two VLists.
         *
         * The logic of the weak bound is the following: the variation between the old
         * VList and the new one is equal to the maximum distance between a ValueFunction
         * in the old VList with its closest match in the new VList. So the farthest from
         * closest.
         *
         * We define distance between two ValueFunctions as the maximum between their
         * element-wise difference.
         *
         * @param oldV The fist VList to compare.
         * @param newV The second VList to compare.
         *
         * @return The weak bound distance between the two arguments.
         */
        double weakBoundDistance(const VList & oldV, const VList & newV);

        bool operator<(const VEntry & lhs, const VEntry & rhs);
        bool operator>(const VEntry & lhs, const VEntry & rhs);

        /**
         * @brief This function generates a random belief uniformly in the space of beliefs.
         *
         * @param S The number of states of the resulting belief.
         * @param generator A random number generator.
         *
         * @return A new random belief.
         */
        template <typename G>
        Belief makeRandomBelief(size_t S, G & generator) {
            static std::uniform_real_distribution<double> sampleDistribution(0.0, 1.0);
            Belief b(S);
            for ( size_t s = 0; s < S; ++s )
                b[s] = sampleDistribution(generator);

            auto sum = b.sum();

            if ( checkEqualSmall(sum, 0.0) ) b[0] = 1.0;
            else b /= sum;

            return b;
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
        template <typename M, typename std::enable_if<is_model<M>::value && !is_model_eigen<M>::value>::type* = nullptr>
        Belief updateBelief(const M & model, const Belief & b, size_t a, size_t o) {
            size_t S = model.getS();
            Belief br(S);

            double totalSum = 0.0;
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    sum += model.getTransitionProbability(s,a,s1) * b[s];

                br[s1] = model.getObservationProbability(s1,a,o) * sum;
                totalSum += br[s1];
            }

            if ( checkEqualSmall(totalSum, 0.0) ) br[0] = 1.0;
            else br /= totalSum;

            return br;
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
        template <typename M, typename std::enable_if<is_model_eigen<M>::value>::type* = nullptr>
        Belief updateBelief(const M & model, const Belief & b, size_t a, size_t o) {
            size_t S = model.getS();
            Belief br(S);

            // col suffers from a bug atm, so we can't use col; we fallback on block.
            br = model.getObservationFunction(a).block(0,o,S,1).transpose().cwiseProduct(Vector::Ones(S).transpose() * model.getTransitionFunction(a).cwiseProduct(b.replicate(1, S))).transpose();
            double totalSum = br.sum();

            if ( checkEqualSmall(totalSum, 0.0) ) br[0] = 1.0;
            else br /= totalSum;

            return br;
        }

        /**
         * @brief This function computes an immediate reward based on a belief rather than a state.
         *
         * @param model The POMDP model to use.
         * @param b The belief to use.
         * @param a The action performed from the belief.
         *
         * @return The immediate reward.
         */
        template <typename M, typename std::enable_if<is_model<M>::value && !is_model_eigen<M>::value>::type* = nullptr>
        double beliefExpectedReward(const M& model, const Belief & b, size_t a) {
            double rew = 0.0; size_t S = model.getS();
            for ( size_t s = 0; s < S; ++s )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    rew += model.getTransitionProbability(s, a, s1) * model.getExpectedReward(s, a, s1) * b[s];

            return rew;
        }

        /**
         * @brief This function computes an immediate reward based on a belief rather than a state.
         *
         * @param model The POMDP model to use.
         * @param b The belief to use.
         * @param a The action performed from the belief.
         *
         * @return The immediate reward.
         */
        template <typename M, typename std::enable_if<is_model_eigen<M>::value>::type* = nullptr>
        double beliefExpectedReward(const M& model, const Belief & b, size_t a) {
            return (model.getTransitionFunction(a).cwiseProduct(model.getRewardFunction(a)) * Vector::Ones(model.getS())).dot(b);
        }

        /**
         * @brief This function computes the probability of obtaining an observation from a belief and action.
         *
         * @param model The POMDP model to use.
         * @param b The belief to start from.
         * @param a The action performed.
         * @param o The observation that should be received.
         *
         * @return The probability of getting the observation from that belief and action.
         */
        template <typename M, typename std::enable_if<is_model<M>::value && !is_model_eigen<M>::value>::type* = nullptr>
        double beliefObservationProbability(const M& model, const Belief & b, size_t a, size_t o) {
            double p = 0.0; size_t S = model.getS();
            // This is basically the same as a belief update, but unnormalized
            // and we sum all elements together..
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    sum += model.getTransitionProbability(s, a, s1) * b[s];

                p += model.getObservationProbability(s1, a, o) * sum;
            }
            return p;
        }

        /**
         * @brief This function computes the probability of obtaining an observation from a belief and action.
         *
         * @param model The POMDP model to use.
         * @param b The belief to start from.
         * @param a The action performed.
         * @param o The observation that should be received.
         *
         * @return The probability of getting the observation from that belief and action.
         */
        template <typename M, typename std::enable_if<is_model_eigen<M>::value>::type* = nullptr>
        double beliefObservationProbability(const M& model, const Belief & b, size_t a, size_t o) {
            return (b.transpose() * model.getTransitionFunction(a) * model.getObservationFunction(a).col(o))(0);
        }

        /**
         * @brief This function returns an iterator pointing to the best value for the specified belief.
         *
         * Ideally I would like to SFINAE that the iterator type is from VList, but at the moment
         * it would take too much time. Just remember that!
         *
         * @tparam Iterator An iterator, can be const or not, from VList.
         * @param bbegin The begin of the belief.
         * @param bend The end of the belief.
         * @param begin The start of the range to look in.
         * @param end The end of the range to look in (excluded).
         * @param value A pointer to double, which gets set to the value of the given belief with the found VEntry.
         *
         * @return An iterator pointing to the best choice in range.
         */
        template <typename Iterator>
        Iterator findBestAtBelief(const Belief & b, Iterator begin, Iterator end, double * value = nullptr) {
            auto bestMatch = begin;
            double bestValue = b.dot(std::get<VALUES>(*bestMatch));

            while ( (++begin) < end ) {
                auto & v = std::get<VALUES>(*begin);
                double currValue = b.dot(v);
                if ( currValue > bestValue || ( currValue == bestValue && ( AIToolbox::operator>(v, std::get<VALUES>(*bestMatch) )) ) ) {
                    bestMatch = begin;
                    bestValue = currValue;
                }
            }
            if ( value ) *value = bestValue;
            return bestMatch;
        }

        /**
         * @brief This function returns an iterator pointing to the best value for the specified corner of the simplex space.
         *
         * Ideally I would like to SFINAE that the iterator type is from VList, but at the moment
         * it would take too much time. Just remember that!
         *
         * @tparam Iterator An iterator, can be const or not, from VList.
         * @param corner The corner of the belief space we are checking.
         * @param begin The start of the range to look in.
         * @param end The end of the range to look in (excluded).
         *
         * @return An iterator pointing to the best choice in range.
         */
        template <typename Iterator>
        Iterator findBestAtSimplexCorner(size_t corner, Iterator begin, Iterator end, double * value = nullptr) {
            auto bestMatch = begin;
            double bestValue = std::get<VALUES>(*bestMatch)[corner];

            while ( (++begin) < end ) {
                auto & v = std::get<VALUES>(*begin);
                double currValue = v[corner];
                if ( currValue > bestValue || ( currValue == bestValue && ( AIToolbox::operator>(v, std::get<VALUES>(*bestMatch)) ) ) ) {
                    bestMatch = begin;
                    bestValue = currValue;
                }
            }
            if ( value ) *value = bestValue;
            return bestMatch;
        }

        /**
         * @brief This function finds and moves the ValueFunction with the highest value for the given belief at the beginning of the specified range.
         *
         * This function uses an already existing bound containing previously marked useful
         * ValueFunctions. The order is 'begin'->'bound'->'end', where bound may be equal to end
         * where no previous bound exists. The found ValueFunction is moved between 'begin' and
         * 'bound', but only if it was not there previously.
         *
         * @tparam Iterator An iterator, can be const or not, from VList.
         * @param bbegin The begin of the belief.
         * @param bend   The end of the belief.
         * @param begin The begin of the search range.
         * @param bound The begin of the 'useful' range.
         * @param end The range end to be checked. It is NOT included in the search.
         *
         * @return The iterator pointing to the element with the highest dot product with the input belief.
         */
        template <typename Iterator>
        Iterator extractWorstAtBelief(const Belief & b, Iterator begin, Iterator bound, Iterator end) {
            auto bestMatch = findBestAtBelief(b, begin, end);

            if ( bestMatch >= bound )
                std::iter_swap(bestMatch, bound++);

            return bound;
        }

        /**
         * @brief This function finds and moves all best ValueFunctions in the simplex corners at the beginning of the specified range.
         *
         * What this function does is to find out which ValueFunctions give the highest value in
         * corner beliefs. Since multiple corners may use the same ValueFunction, the number of
         * found ValueFunctions may not be the same as the number of corners.
         *
         * This function uses an already existing bound containing previously marked useful
         * ValueFunctions. The order is 'begin'->'bound'->'end', where bound may be equal to end
         * where no previous bound exists. All found ValueFunctions are added between 'begin' and
         * 'bound', but only if they were not there previously.
         *
         * @param S The number of corners of the simplex.
         * @param begin The begin of the search range.
         * @param bound The begin of the 'useful' range.
         * @param end The end of the search range. It is NOT included in the search.
         *
         * @return The new bound iterator.
         */
        template <typename Iterator>
        Iterator extractWorstAtSimplexCorners(size_t S, Iterator begin, Iterator bound, Iterator end) {
            if ( end == bound ) return bound;

            // For each corner
            for ( size_t s = 0; s < S; ++s ) {
                auto bestMatch = findBestAtSimplexCorner(s, begin, end);

                if ( bestMatch >= bound )
                    std::iter_swap(bestMatch, bound++);
            }
            return bound;
        }

        /**
         * @brief This function finds and movess all ValueFunctions in the VList that are dominated by others.
         *
         * This function performs simple comparisons between all ValueFunctions in the VList,
         * and is thus much more performant than a full-fledged prune, since that would need to solve
         * multiple linear programming problems. However, this function will not return the truly
         * parsimonious set of ValueFunctions, as its pruning powers are limited.
         *
         * Dominated elements will be moved at the end of the range for safe removal.
         *
         * @param S The number of states in the Model.
         * @param begin The begin of the list that needs to be pruned.
         * @param begin The end of the list that needs to be pruned.
         *
         * @return The iterator that separates dominated elements with non-pruned.
         */
        template <typename Iterator>
        Iterator extractDominated(size_t S, Iterator begin, Iterator end) {
            if ( std::distance(begin, end) < 2 ) return end;

            // We use this comparison operator to filter all dominated vectors.
            // We define a vector to be dominated by an equal vector, so that
            // we can remove duplicates in a single swoop. However, we avoid
            // removing everything by returning false for comparison of the vector with itself.
            struct {
                const MDP::Values * rhs;
                size_t S;
                bool operator()(const VEntry & lhs) {
                    auto & lhsV = std::get<VALUES>(lhs);
                    if ( &(lhsV) == rhs ) return false;
                    for ( size_t i = 0; i < S; ++i )
                        if ( (*rhs)[i] > lhsV[i] ) return false;
                    return true;
                }
            } dominates;

            dominates.S = S;

            // For each vector, if we find a vector that dominates it, then we remove it.
            // Otherwise we continue, comparing every vector with every other non-dominated
            // vector.
            Iterator iter = begin, helper;
            while ( iter < end ) {
                dominates.rhs = &(std::get<VALUES>(*iter));
                helper = std::find_if(begin, end, dominates);
                if ( helper != end )
                    std::iter_swap( iter, --end );
                else
                    ++iter;
            }
            return end;
        }

    }
}

#endif
