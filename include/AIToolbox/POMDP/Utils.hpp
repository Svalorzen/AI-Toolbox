#ifndef AI_TOOLBOX_POMDP_UTILS_HEADER_FILE
#define AI_TOOLBOX_POMDP_UTILS_HEADER_FILE

#include <cstddef>
#include <iterator>
#include <numeric>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Probability.hpp>
#include <AIToolbox/Utils/Polytope.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>

#include <boost/functional/hash.hpp>

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
    // This implementation is temporary until we can substitute both with the
    // spaceship operator (<=>) in C++20.
    bool operator==(const VEntry & lhs, const VEntry & rhs);

    /**
     * @brief This function enables hashing of VEntries with boost::hash.
     */
    inline size_t hash_value(const VEntry & v) {
        size_t seed = 0;
        boost::hash_combine(seed, v.action);
        boost::hash_combine(seed, v.observations);
        boost::hash_combine(seed, v.values);
        return seed;
    }

    /**
     * @brief This function is used as iterator projection to obtain the Values of a VEntry.
     */
    inline const MDP::Values & unwrap(const VEntry & ve) {
        return ve.values;
    }

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
     * @brief This function creates the SOSA matrix for the input POMDP.
     *
     * The SOSA matrix is a way to represent the observation and transition
     * functions in a single function, at the same time.
     *
     * Each cell in this four-dimensional matrix contains the probability of
     * getting to state s' while obtaining observation o when starting with
     * state s and action a.
     *
     * This matrix is less space-efficient than storing both matrices separately,
     * but it can save you some time if you need its values multiple times in a
     * loop (for example in the FastInformedBound algorithm).
     *
     * @param m The input POMDP to extract the SOSA matrix from.
     *
     * @return The SOSA matrix for the input pomdp.
     */
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    auto makeSOSA(const M & m) {
        if constexpr(is_model_eigen_v<M>) {
            boost::multi_array<remove_cv_ref_t<decltype(m.getTransitionFunction(0))>, 2> retval( boost::extents[m.getA()][m.getO()] );
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    void updateBeliefUnnormalized(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        if constexpr(is_model_eigen_v<M>) {
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    void updateBeliefPartial(const M & model, const Belief & b, const size_t a, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        if constexpr(is_model_eigen_v<M>) {
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    void updateBeliefPartialUnnormalized(const M & model, const Belief & b, const size_t a, const size_t o, Belief * bRet) {
        if (!bRet) return;

        auto & br = *bRet;

        if constexpr(is_model_eigen_v<M>) {
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
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
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    double beliefExpectedReward(const M& model, const Belief & b, const size_t a) {
        if constexpr (is_model_eigen_v<M>) {
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
     * @brief This function computes the best VEntry for the input belief from the input VLists.
     *
     * This function computes the best alphavector for the input belief. It
     * assumes as input a list of VLists, one per observation. You can produce
     * them with the Projecter class, for example.
     *
     * For each observation it will select the match with the best value, and
     * use it to compute the output VEntry.
     *
     * The action is not used to perform computations here, but it is fed
     * directly into the returned VEntry.
     *
     * @tparam ActionRow The type of the list of VLists.
     * @param b The belief to compute the VEntry for.
     * @param row The list of VLists, one per observation.
     * @param a The action this Ventry stands for.
     * @param value A pointer to double, which gets set to the value of the given belief with the generated VEntry.
     *
     * @return The best VEntry for the input belief.
     */
    template <typename ActionRow>
    VEntry crossSumBestAtBelief(const Belief & b, const ActionRow & row, const size_t a, double * value = nullptr) {
        const size_t O = row.size();
        VEntry entry(b.size(), a, O);
        double v = 0.0, tmp;

        // We compute the crossSum between each best vector for the belief.
        for ( size_t o = 0; o < O; ++o ) {
            const auto & r = row[o];
            auto begin = std::begin(r);
            auto end   = std::end(r);

            auto bestMatch = findBestAtPoint(b, begin, end, &tmp, unwrap).base();

            entry.values += bestMatch->values;
            v += tmp;

            entry.observations[o] = bestMatch->observations[0];
        }
        if (value) *value = v;
        return entry;
    }

    /**
     * @brief This function computes the best VEntry for the input belief across all actions.
     *
     * This function needs the projections of the previous timestep's VLists in
     * order to work. It will then compute the best VEntry for the input belief
     * across all actions.
     *
     * @tparam Projections The type of the 2D array of VLists containing all the projections.
     * @param b The belief to compute the VEntry for.
     * @param projs The projections of the old VLists.
     * @param value A pointer to double, which gets set to the value of the given belief with the generated VEntry.
     *
     * @return The best VEntry for the input belief.
     */
    template <typename Projections>
    VEntry crossSumBestAtBelief(const Belief & b, const Projections & projs, double * value = nullptr) {
        const size_t A = projs.size();
        VEntry entry;

        double bestValue = std::numeric_limits<double>::lowest(), tmp;
        for ( size_t a = 0; a < A; ++a ) {
            auto result = crossSumBestAtBelief(b, projs[a], a, &tmp);
            if (tmp > bestValue) {
                bestValue = tmp;
                std::swap(entry, result);
            }
        }
        if (value) *value = bestValue;
        return entry;
    }

    /**
     * @brief This function obtains the best action with respect to the input VList.
     *
     * This function pretty much does what the Projecter class does. The
     * difference is that while the Projecter expands one step in the future
     * every single entry in the input VList, this only does it to the input
     * Belief.
     *
     * This allows to both avoid a lot of work if we wouldn't need to reuse the
     * Projecter results a lot, and simplifies the crossSum step.
     *
     * @param pomdp The model to use.
     * @param immediateRewards The immediate rewards of the model.
     * @param initialBelief The belief where the best action needs to be found.
     * @param lbVList The alphavectors to use.
     * @param alpha Optionally, the output alphavector for the best action. Does not need preallocation.
     *
     * @return The best action in the input belief with respect to the input VList.
     */
    template <typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    std::tuple<size_t, double> bestConservativeAction(const M & pomdp, MDP::QFunction immediateRewards, const Belief & initialBelief, const VList & lbVList, MDP::Values * alpha = nullptr) {
        // Note that we update inline the alphavectors in immediateRewards
        Vector bpAlpha(pomdp.getS());
        // Storage to avoid reallocations
        Belief intermediateBelief(pomdp.getS());
        Belief nextBelief(pomdp.getS());

        for (size_t a = 0; a < pomdp.getA(); ++a) {
            updateBeliefPartial(pomdp, initialBelief, a, &intermediateBelief);

            bpAlpha.setZero();

            for (size_t o = 0; o < pomdp.getO(); ++o) {
                updateBeliefPartialUnnormalized(pomdp, intermediateBelief, a, o, &nextBelief);

                const auto nextBeliefProbability = nextBelief.sum();
                if (checkEqualSmall(nextBeliefProbability, 0.0)) continue;
                // Now normalized
                nextBelief /= nextBeliefProbability;

                const auto it = findBestAtPoint(nextBelief, std::begin(lbVList), std::end(lbVList), nullptr, unwrap);

                bpAlpha += pomdp.getObservationFunction(a).col(o).cwiseProduct(it->values);
            }
            immediateRewards.col(a) += pomdp.getDiscount() * pomdp.getTransitionFunction(a) * bpAlpha;
        }

        size_t id;
        double v = (initialBelief.transpose() * immediateRewards).maxCoeff(&id);

        // Copy alphavector for selected action if needed
        if (alpha) *alpha = immediateRewards.col(id);

        return std::make_tuple(id, v);
    }

    /**
     * @brief This function obtains the best action with respect to the input QFunction and UbV.
     *
     * This function simply computes the upper bound for all beliefs that can
     * be reached from the input belief. For each action, their values are
     * summed (after multiplying each by the probability of it happening), and
     * the best action extracted.
     *
     * @tparam useLP Whether we want to use LP interpolation, rather than sawtooth. Defaults to true.
     * @param pomdp The model to look the action for.
     * @param immediateRewards The immediate rewards of the model.
     * @param belief The belief to find the best action in.
     * @param ubQ The current QFunction for this model.
     * @param ubV The current list of belief/values for this model.
     * @param vals Optionally, an output vector containing the per-action upper-bound values. Does not need preallocation, and passing it does not result in more work.
     *
     * @return The best action-value pair.
     */
    template <bool useLP = true, typename M, std::enable_if_t<is_model_v<M>, int> = 0>
    std::tuple<size_t, double> bestPromisingAction(const M & pomdp, const MDP::QFunction & immediateRewards, const Belief & belief, const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV, Vector * vals = nullptr) {
        Vector storage;
        Vector & qvals = vals ? *vals : storage;

        qvals = belief.transpose() * immediateRewards;

        // Storage to avoid reallocations
        Belief intermediateBelief(pomdp.getS());
        Belief nextBelief(pomdp.getS());

        for (size_t a = 0; a < pomdp.getA(); ++a) {
            updateBeliefPartial(pomdp, belief, a, &intermediateBelief);
            double sum = 0.0;
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                updateBeliefPartialUnnormalized(pomdp, intermediateBelief, a, o, &nextBelief);

                const auto prob = nextBelief.sum();
                if (checkEqualSmall(prob, 0.0)) continue;
                // Note that we do not normalize nextBelief since we'd also
                // have to multiply the result by the same probability. Instead
                // we don't normalize, and we don't multiply, so we save some
                // work.
                if constexpr (useLP)
                    sum += std::get<0>(LPInterpolation(nextBelief, ubQ, ubV));
                else
                    sum += std::get<0>(sawtoothInterpolation(nextBelief, ubQ, ubV));
            }
            qvals[a] += pomdp.getDiscount() * sum;
        }
        size_t bestAction;
        double bestValue = qvals.maxCoeff(&bestAction);

        return std::make_tuple(bestAction, bestValue);
    }
}

#endif
