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
     * @brief This function creates a default ValueFunction.
     *
     * The default ValueFunction contains a single VList with inside a single
     * VEntry: do action 0, with all value zeroes.
     *
     * The VList is a necessary byproduct that is needed when computing the
     * whole ValueFunction recursively via dynamic programming.
     *
     * In the end, to act, it's not needed, but it's probably more hassle to
     * remove the entry, and so we leave it there. So in general we always
     * assume it's there.
     *
     * Another peculiarity of the default VEntry is that it's the only place
     * where the observation id vector is empty, since nobody is ever supposed
     * to go looking in there.
     *
     * @param S The number of possible states.
     *
     * @return A new ValueFunction.
     */
    ValueFunction makeValueFunction(size_t S);

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
     * @brief This function creates the SOSA table for the input POMDP.
     *
     * The SOSA table is a way to represent the observation and transition
     * functions in a single function, at the same time.
     *
     * Each cell in this four-dimensional table contains the probability of
     * getting to state s' while obtaining observation o when starting with
     * state s and action a.
     *
     * This table is less space-efficient than storing both tables separately,
     * but it can save you some time if you need its values multiple times in a
     * loop (for example in the FastInformedBound algorithm).
     *
     * @param m The input POMDP to extract the SOSA table from.
     *
     * @return The SOSA table for the input pomdp.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    auto makeSOSA(const M & m) {
        if constexpr(is_model_eigen<M>::value) {
            boost::multi_array<typename remove_cv_ref<decltype(m.getTransitionFunction(0))>::type, 2> retval( boost::extents[m.getA()][m.getO()] );
            for (size_t a = 0; a < m.getA(); ++a)
                for (size_t o = 0; o < m.getO(); ++o)
                    retval[a][o] = m.getTransitionFunction(a) * Vector(m.getObservationFunction(a).col(o)).asDiagonal();
            return retval;
        } else {
            Matrix4D retval( boost::extents[m.getA()][m.getO()] );
            for (size_t a = 0; a < m.getA(); ++a) {
                for (size_t o = 0; o < m.getO(); ++o) {
                    retval[a][o].resize(m.getS(), m.getS());
                    for (size_t s = 0; s < m.getS(); ++s)
                        for (size_t s1 = 0; s1 < m.getS(); ++s1)
                            retval[a][o](s, s1) = m.getTransitionProbability(s, a, s1) * m.getObservationProbability(s1, a, o);
                }
            }
            return retval;
        }
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
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
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
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
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
     * NOTE: This function assumes that the update and the normalization are
     * possible, i.e. that from the input belief and action it is possible to
     * receive the input observation.
     *
     * If that cannot be guaranteed, use the updateBeliefUnnormalized()
     * function and do the normalization yourself (and check for it).
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    void updateBelief(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        updateBeliefUnnormalized(model, b, a, o, bRet);

        auto & br = *bRet;
        br /= br.sum();
    }

    /**
     * @brief Creates a new belief reflecting changes after an action and observation for a particular Model.
     *
     * This function needs to create a new belief since modifying a belief
     * in place is not possible. This is because each cell update for the
     * new belief requires all values from the previous belief.
     *
     * NOTE: This function assumes that the update and the normalization are
     * possible, i.e. that from the input belief and action it is possible to
     * receive the input observation.
     *
     * If that cannot be guaranteed, use the updateBeliefUnnormalized()
     * function and do the normalization yourself (and check for it).
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    Belief updateBelief(const M & model, const Belief & b, const size_t a, const size_t o) {
        Belief br(model.getS());
        updateBelief(model, b, a, o, &br);
        return br;
    }

    /**
     * @brief This function partially updates a belief.
     *
     * This function is useful in case one needs to update a belief for all
     * possible observations. In such a case, it is possible to avoid repeating
     * the same operations by creating an intermediate belief, that only
     * depends on the action and not on the observation.
     *
     * From this intermediate result it will be then possible to obtain the end
     * belief by supplying the same action and the desired observation.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     * @param bRet The output belief.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    void updateBeliefPartial(const M & model, const Belief & b, const size_t a, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        if constexpr(is_model_eigen<M>::value) {
            br = (b.transpose() * model.getTransitionFunction(a)).transpose();
        } else {
            const size_t S = model.getS();
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                br[s1] = 0.0;
                for ( size_t s = 0; s < S; ++s )
                    br[s1] += model.getTransitionProbability(s,a,s1) * b[s];
            }
        }
    }

    /**
     * @brief This function partially updates a belief.
     *
     * This function is useful in case one needs to update a belief for all
     * possible observations. In such a case, it is possible to avoid repeating
     * the same operations by creating an intermediate belief, that only
     * depends on the action and not on the observation.
     *
     * From this intermediate result it will be then possible to obtain the end
     * belief by supplying the same action and the desired observation.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The old belief.
     * @param a The action taken during the transition.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    Belief updateBeliefPartial(const M & model, const Belief & b, const size_t a) {
        Belief bRet(model.getS());
        updateBeliefPartial(model, b, a, &bRet);
        return bRet;
    }

    /**
     * @brief This function terminates the unnormalized update of a partially updated belief.
     *
     * This function is useful in case one needs to update a belief for all
     * possible observations. In such a case, it is possible to avoid repeating
     * the same operations by creating an intermediate belief, that only
     * depends on the action and not on the observation.
     *
     * \sa updateBeliefPartial
     *
     * Note that the input action here must be the same one that produced the
     * intermediate result.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The intermediate belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    void updateBeliefPartialUnnormalized(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        if constexpr(is_model_eigen<M>::value) {
            br = model.getObservationFunction(a).col(o).cwiseProduct(b);
        } else {
            const size_t S = model.getS();
            for ( size_t s = 0; s < S; ++s )
                br[s] = model.getObservationProbability(s, a, o) * b[s];
        }
    }

    /**
     * @brief This function terminates the unnormalized update of a partially updated belief.
     *
     * This function is useful in case one needs to update a belief for all
     * possible observations. In such a case, it is possible to avoid repeating
     * the same operations by creating an intermediate belief, that only
     * depends on the action and not on the observation.
     *
     * \sa updateBeliefPartial
     *
     * Note that the input action here must be the same one that produced the
     * intermediate result.
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The intermediate belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    Belief updateBeliefPartialUnnormalized(const M & model, const Belief & b, const size_t a, const size_t o) {
        Belief bRet(model.getS());
        updateBeliefPartialUnnormalized(model, b, a, o, &bRet);
        return bRet;
    }

    /**
     * @brief This function terminates the normalized update of a partially updated belief.
     *
     * This function is useful in case one needs to update a belief for all
     * possible observations. In such a case, it is possible to avoid repeating
     * the same operations by creating an intermediate belief, that only
     * depends on the action and not on the observation.
     *
     * \sa updateBeliefPartial
     *
     * Note that the input action here must be the same one that produced the
     * intermediate result.
     *
     * NOTE: This function assumes that the update and the normalization are
     * possible, i.e. that from the input belief and action it is possible to
     * receive the input observation.
     *
     * If that cannot be guaranteed, use the updateBeliefUnnormalized()
     * function and do the normalization yourself (and check for it).
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The intermediate belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     * @param bRet The output belief.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    void updateBeliefPartialNormalized(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        updateBeliefPartialUnnormalized(model, b, a, o, bRet);

        br /= br.sum();
    }

    /**
     * @brief This function terminates the normalized update of a partially updated belief.
     *
     * This function is useful in case one needs to update a belief for all
     * possible observations. In such a case, it is possible to avoid repeating
     * the same operations by creating an intermediate belief, that only
     * depends on the action and not on the observation.
     *
     * \sa updateBeliefPartial
     *
     * Note that the input action here must be the same one that produced the
     * intermediate result.
     *
     * NOTE: This function assumes that the update and the normalization are
     * possible, i.e. that from the input belief and action it is possible to
     * receive the input observation.
     *
     * If that cannot be guaranteed, use the updateBeliefUnnormalized()
     * function and do the normalization yourself (and check for it).
     *
     * @tparam M The type of the POMDP Model.
     * @param model The model used to update the belief.
     * @param b The intermediate belief.
     * @param a The action taken during the transition.
     * @param o The observation registered.
     */
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    Belief updateBeliefPartialNormalized(const M & model, const Belief & b, const size_t a, const size_t o) {
        auto newB = updateBeliefPartialUnnormalized(model, b, a, o);
        newB /= newB.sum();
        return newB;
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
    template <typename M, std::enable_if_t<is_model<M>::value, int> = 0>
    double beliefExpectedReward(const M& model, const Belief & b, const size_t a) {
        if constexpr (is_model_eigen<M>::value) {
            return model.getRewardFunction().col(a).dot(b);
        } else {
            double rew = 0.0; const size_t S = model.getS();
            for ( size_t s = 0; s < S; ++s )
                for ( size_t s1 = 0; s1 < S; ++s1 )
                    rew += model.getTransitionProbability(s, a, s1) * model.getExpectedReward(s, a, s1) * b[s];

            return rew;
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
     * When multiple beliefs support the same VEntry, the ones with the best
     * values are returned.
     *
     * The input VEntries may contain elements which are not supported by any
     * of the input Beliefs (although if they exist they may slow down the
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
    BIterator extractBestUsefulBeliefs(BIterator bbegin, BIterator bend, VIterator begin, VIterator end) {
        const auto beliefsN = std::distance(bbegin, bend);
        const auto entriesN = std::distance(begin, end);

        std::vector<std::pair<BIterator, double>> bestValues(entriesN, {bend, std::numeric_limits<double>::lowest()});
        const auto maxBound = beliefsN < entriesN ? bend : bbegin + entriesN;

        // So the idea here is that we advance IT only if we found a belief
        // which supports a previously unsupported VEntry. This allows us to
        // avoid doing later work for compacting the beliefs before the bound.
        //
        // If instead the found belief takes into consideration an already
        // supported VEntry, then it either is better or not. If it's better,
        // we swap it with whatever was before. In both cases, the belief to
        // discard ends up at the end and we decrease the bound.
        auto it = bbegin;
        auto bound = bend;
        while (it < bound && it < maxBound) {
            double value;
            const auto vId = std::distance(begin, findBestAtBelief(*it, begin, end, &value));
            if (bestValues[vId].second < value) {
                if (bestValues[vId].first == bend) {
                    bestValues[vId] = {it++, value};
                    continue;
                } else {
                    bestValues[vId].second = value;
                    std::iter_swap(bestValues[vId].first, it);
                }
            }
            std::iter_swap(it, --bound);
        }
        if (it == bound) return it;

        // If all VEntries have been supported by at least one belief, then we
        // can finish up the rest with less swaps and checks. Here we only swap
        // with the best if needed, otherwise we don't have to do anything.
        //
        // This is because we can return one belief per VEntry at the most, so
        // if we're here the bound is not going to move anyway.
        while (it < bound) {
            double value;
            const auto vId = std::distance(begin, findBestAtBelief(*it, begin, end, &value));
            if (bestValues[vId].second < value) {
                bestValues[vId].second = value;
                std::iter_swap(bestValues[vId].first, it);
            }
            ++it;
        }
        return maxBound;
    }
}

#endif
