#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <cstddef>
#include <iterator>
#include <numeric>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This function lexicographically sorts VEntries.
     *
     * This is useful during testing in order to sort and compare a solution
     * with the correct result.
     *
     * @param lhs The left hand side of the comparison.
     * @param rhs The right hand side of the comparison.
     *
     * @return True if lhs is less than rhs lexicographically, false otherwise.
     */
    bool operator<(const VEntry & lhs, const VEntry & rhs);

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
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    Belief updateBelief(const M & model, const Belief & b, const size_t a, const size_t o) {
        Belief br(model.getS());
        updateBelief(model, b, a, o, &br);
        return br;
    }

    /**
     * @brief Creates a new belief reflecting changes after an action and observation for a particular Model.
     *
     * This function writes directly into the provided Belief pointer. It assumes
     * that the pointer points to a correctly sized Belief. It does a basic nullptr
     * check.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    void updateBelief(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        updateBeliefUnnormalized(model, b, a, o, bRet);

        auto & br = *bRet;
        const double totalSum = br.sum();

        if ( checkEqualSmall(totalSum, 0.0) ) br[0] = 1.0;
        else br /= totalSum;
    }

    /**
     * @brief Creates a new belief reflecting changes after an action and observation for a particular Model.
     *
     * This function needs to create a new belief since modifying a belief
     * in place is not possible. This is because each cell update for the
     * new belief requires all values from the previous belief.
     *
     * This function will not normalize the output, nor is guaranteed
     * to return a non-completely-zero vector.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     */
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    Belief updateBeliefUnnormalized(const M & model, const Belief & b, const size_t a, const size_t o) {
        Belief br(model.getS());
        updateBeliefUnnormalized(model, b, a, o, &br);
        return br;
    }

    /**
     * @brief Creates a new belief reflecting changes after an action and observation for a particular Model.
     *
     * This function writes directly into the provided Belief pointer. It assumes
     * that the pointer points to a correctly sized Belief. It does a basic nullptr
     * check.
     *
     * This function will not normalize the output, nor is guaranteed
     * to return a non-completely-zero vector.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    void updateBeliefUnnormalized(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        if constexpr(is_model_eigen<M>::value) {
            br = model.getObservationFunction(a).col(o).cwiseProduct((b.transpose() * model.getTransitionFunction(a)).transpose());
        } else {
            const size_t S = model.getS();
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                double sum = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    sum += model.getTransitionProbability(s,a,s1) * b[s];

                br[s1] = model.getObservationProbability(s1,a,o) * sum;
            }
        }
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
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    double beliefExpectedReward(const M& model, const Belief & b, const size_t a) {
        if constexpr (is_model_eigen<M>::value) {
            return (model.getTransitionFunction(a).cwiseProduct(model.getRewardFunction(a)) * Vector::Ones(model.getS())).dot(b);
        } else {
            double rew = 0.0; const size_t S = model.getS();
            for ( size_t s = 0; s < S; ++s )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    rew += model.getTransitionProbability(s, a, s1) * model.getExpectedReward(s, a, s1) * b[s];

            return rew;
        }
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
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    double beliefObservationProbability(const M& model, const Belief & b, const size_t a, const size_t o) {
        if constexpr (is_model_eigen<M>::value) {
            return (b.transpose() * model.getTransitionFunction(a) * model.getObservationFunction(a).col(o))(0);
        } else {
            double p = 0.0; const size_t S = model.getS();
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
            const double currValue = b.dot(v);
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
    Iterator findBestAtSimplexCorner(const size_t corner, Iterator begin, Iterator end, double * value = nullptr) {
        auto bestMatch = begin;
        double bestValue = std::get<VALUES>(*bestMatch)[corner];

        while ( (++begin) < end ) {
            auto & v = std::get<VALUES>(*begin);
            const double currValue = v[corner];
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
     * @param bend The end of the belief.
     * @param begin The begin of the search range.
     * @param bound The begin of the 'useful' range.
     * @param end The range end to be checked. It is NOT included in the search.
     *
     * @return The new bound iterator.
     */
    template <typename Iterator>
    Iterator extractBestAtBelief(const Belief & b, Iterator begin, Iterator bound, Iterator end) {
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
    Iterator extractBestAtSimplexCorners(const size_t S, Iterator begin, Iterator bound, Iterator end) {
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
     * @brief This function finds and moves all non-useful beliefs at the end of the input range.
     *
     * This function helps remove beliefs which do not support any VEntry and
     * are thus not useful for improving the VList bounds.
     *
     * This function moves all non-useful beliefs at the end of the input
     * range, and returns the resulting iterator pointing to the first
     * non-useful belief.
     *
     * Note that this function will shuffle around the VEntries, as it needs to
     * keep track of which VEntries have already been supported by some beliefs
     * (so that if another beliefs supports them, we know it is not useful).
     *
     * The input VEntries may contain elements which are not supported by any
     * of the input Beliefs (although if they exist they will slow down the
     * function).
     *
     * @param it The beginning of the belief range to check.
     * @param bend The end of the belief range to check.
     * @param begin The beginning of the VEntry range to check against.
     * @param end The end of the VEntry range to check against.
     *
     * @return An iterator pointing to the first non-useful belief.
     */
    template <typename BIterator, typename VIterator>
    BIterator extractUsefulBeliefs(BIterator it, BIterator bend, VIterator begin, VIterator end) {
        auto bound = begin;
        // We stop if we looked at all beliefs, or if there's no VEntries to
        // support anymore.
        while (it < bend && bound < end) {
            const auto newBound = extractBestAtBelief(*it, begin, bound, end);
            if (bound == newBound) {
                std::iter_swap(it, --bend);
            } else {
                bound = newBound;
                ++it;
            }
        }
        return it;
    }
}

#endif
