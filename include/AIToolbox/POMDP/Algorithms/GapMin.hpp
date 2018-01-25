#ifndef AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE
#define AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE

#include <algorithm>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/POMDP/Types.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>
#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the GapMin algorithm.
     */
    class GapMin {
        public:
            /**
             * @brief Basic constructor.
             */
            GapMin();

            /**
             * @brief This function solves a POMDP::Model approximately.
             */
            template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
            std::tuple<double, ValueFunction> operator()(const M & model, const Belief & initialBelief);

        private:

            using UbVType = std::pair<std::vector<Belief>, std::vector<double>>;
            class GapTupleLess;

            template <typename M, typename std::enable_if<is_model<M>::value, int>::type = 0>
            std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> selectReachableBeliefs(
                const M & model,
                const Belief &,
                const VList &,
                const std::vector<Belief> &,
                const MDP::QFunction &, const UbVType &
            );

            double epsilon_;
    };

    template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
    std::tuple<double, ValueFunction> GapMin::operator()(const M & pomdp, const Belief & initialBelief) {
        constexpr unsigned infiniteHorizon = 1000000;

        // Helper methods
        BlindStrategies bs(infiniteHorizon, epsilon_);
        FastInformedBound fib(infiniteHorizon, epsilon_);
        PBVI pbvi(0, infiniteHorizon, epsilon_);

        // Here we use the BlindStrategies in order to obtain a very simple
        // initial lower bound.
        auto lbVList = bs(pomdp, true);
        {
            const auto unwrap = +[](VEntry & ve) -> MDP::Values & {return std::get<VALUES>(ve);};
            const auto rbegin = boost::make_transform_iterator(std::begin(lbVList), unwrap);
            const auto rend   = boost::make_transform_iterator(std::end  (lbVList), unwrap);

            lbVList.erase(extractDominated(pomdp.getS(), rbegin, rend).base(), std::end(lbVList));
        }
        auto lbBeliefs = std::vector<Belief>{initialBelief};

        // The same we do here with FIB for the input POMDP.
        auto ubQ = fib(pomdp);

        // At the same time, we start initializing fibQ, which will be our
        // pseudo-alphaVector storage for our belief-POMDPs which we'll create
        // later.
        //
        // The basic idea is to create a new POMDP where each state is a belief
        // of the input POMDP. This allows us to obtain better upper bounds,
        // and project them to our input POMDP.
        auto fibQ = Matrix2D(pomdp.getS()+1, pomdp.getA());
        fibQ.block(0, 0, pomdp.getS(), pomdp.getA()).noalias() = ubQ;
        fibQ.row(pomdp.getS()).noalias() = ubQ * initialBelief;

        // While we store the lower bound as alphaVectors, the upper bound is
        // composed by both alphaVectors (albeit only S of them - out of the
        // FastInformedBound), and a series of belief-value pairs, which we'll
        // use with the later-constructed new POMDP in order to improve our
        // bounds.
        UbVType ubV = {
            {initialBelief}, {fibQ.row(pomdp.getS()).maxCoeff()}
        };

        // We also store two numbers for the overall lowerBound/upperBound
        // differences. They are the values of the lowerBound and the
        // upperBound at the initial belief.
        double lb;
        findBestAtBelief(initialBelief, std::begin(lbVList), std::end(lbVList), &lb);
        double ub = ubV.second[0];

        auto var = ub - lb;
        while (var >= epsilon_) {
            // Now we find beliefs for both lower and upper bound where we
            // think we can improve. For the ub beliefs we also return their
            // values, since we need them to improve the ub.
            auto [newLbBeliefs, newUbBeliefs, newUbVals] = selectReachableBeliefs(pomdp, initialBelief, lbVList, ubQ, ubV);
            const auto newLbBeliefsSize = newLbBeliefs.size();
            const auto newUbBeliefsSize = newUbBeliefs.size();

            if (newLbBeliefsSize > 0) {
                // If we found something interesting for the lower bound, we
                // add it to the beliefs we already had, and we rerun PBVI.
                std::move(std::begin(newLbBeliefs), std::end(newLbBeliefs), std::back_inserter(lbBeliefs));

                // Then we remove all beliefs which don't actively support any
                // alphaVectors.
                lbVList = std::move(pbvi(lbBeliefs, VFunction(lbVList)).back());
                lbBeliefs.erase(extractUsefulBeliefs(
                    std::begin(lbBeliefs), std::end(lbBeliefs),
                    std::begin(lbVList), std::end(lbVList)),
                    std::end(lbBeliefs)
                );

                // And we recompute the lower bound.
                findBestAtBelief(initialBelief, std::begin(lbVList), std::end(lbVList), &lb);
            }

            if (newUbBeliefsSize > 0) {
                // Here we do the same for the upper bound.
                const auto prevRows = pomdp.getS() + ubV.first.size();
                fibQ.conservativeResize(prevRows + newUbBeliefsSize, Eigen::NoChange);

                // For each newly found belief which can improve the upper
                // bound, we add it to to the list containing the beliefs for
                // the upper bound. At the same time we add horizontal planes
                // in the fibQ which will come useful on the next round of
                // FastInformedBound.
                for (size_t i = 0; i < newUbBeliefs.size(); ++i) {
                    ubV.first.emplace_back(std::move(newUbBeliefs[i]));
                    ubV.second.emplace_back(newUbVals[i]);
                    fibQ.row(prevRows + i).fill(newUbVals[i]);
                }

                // We create a new POMDP where each state is a belief.
                auto [newPOMDP, newPOMDPSOSA] = makeNewPomdp(pomdp, ubQ, ubV);
                // And we approximate its upper bound.
                std::tie(std::ignore, fibQ) = fib(newPOMDP, newPOMDPSOSA, std::move(fibQ));

                // We extract from the found upper bound the part for the
                // states of the input POMDP, and we copy them to our
                // upperBound alphavectors. We additionally update the values
                // for all ub beliefs.
                ubQ.noalias() = fibQ.block(0, 0, pomdp.getS(), pomdp.getA());
                for (size_t i = 0; i < ubV.second.size(); ++i)
                    ubV.second[i] = fibQ.row(pomdp.getS() + i).maxCoeff();

                // Finally, we remove some unused stuff, and we recompute the upperbound.
                cleanUp(ubQ, ubV, fibQ);
                ub = upperBound(initialBelief, ubQ, ubV);
            }

            // Stop if we didn't find anything new.
            if (newLbBeliefsSize + newUbBeliefsSize == 0)
                break;

            // Otherwise see what is the difference between upper and lower
            // bound, and let the loop condition check whether we want to stop.
            var = ub - lb;
        }
        // FIXME
        // return ???
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
     * @param initialBelief The belief where the best action needs to be found.
     * @param lbVList The alphavectors to use.
     *
     * @return The best action in the input belief with respect to the input VList.
     */
    template <typename M, typename std::enable_if<is_model<M>::value>::type* = nullptr>
    std::tuple<size_t, double> bestConservativeAction(const M & pomdp, const Belief & initialBelief, const VList & lbVList) {
        auto ir = computeImmediateRewards(pomdp);

        for (size_t a = 0; a < pomdp.getA(); ++a) {
            const Belief intermediateBelief = updateBeliefPartial(pomdp, initialBelief, a);

            Vector bpAlpha(pomdp.getS());
            bpAlpha.fill(0.0);

            for (size_t o = 0; o < pomdp.getO(); ++o) {
                Belief nextBelief = updateBeliefPartialUnnormalized(pomdp, initialBelief, a, o);

                const auto nextBeliefProbability = nextBelief.sum();
                if (checkEqualSmall(nextBeliefProbability, 0.0)) continue;
                // Now normalized
                nextBelief /= nextBeliefProbability;

                auto it = findBestAtBelief(nextBelief, std::begin(lbVList), std::end(lbVList));

                bpAlpha += pomdp.getTransitionFunction(a).col(o).cwiseProduct(*it);
            }
            ir.col(a) += pomdp.getDiscount() * pomdp.getTransitionFunction(a) * bpAlpha;
        }

        size_t id;
        double v = (initialBelief * ir).maxCoeff(&id);

        return std::make_tuple(id, v);
    }

    template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
    std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> GapMin::selectReachableBeliefs(const M & pomdp, const Belief & initialBelief, const VList & lbVList, const std::vector<Belief> & lbBeliefs, const MDP::QFunction & ubQ, const UbVType & ubV) {
        // Queue sorted by gap.         belief,    gap,   prob,    depth,       path
        using QueueElement = std::tuple<Belief, double, double, unsigned, std::vector<Belief>>;
        using QueueType = boost::heap::fibonacci_heap<QueueElement, boost::heap::compare<GapTupleLess>>;

        std::vector<Belief> newLbBeliefs, newUbBeliefs, visitedBeliefs;
        std::vector<double> newUbValues;

        constexpr size_t maxVisitedBeliefs = 1000;
        size_t overwriteCounter = 0;
        visitedBeliefs.reserve(maxVisitedBeliefs);

        QueueType queue;
        unsigned newBeliefs = 0;

        // From the original code, a limitation on how many new beliefs we find.
        const auto maxNewBeliefs = std::max(20lu, (ubV.first.size() + lbVList.size()) / 5lu);

        // We initialize the queue with the initial belief.
        queue.push_back(initialBelief, 0.0, 1.0, 1, {});

        while (!queue.isEmpty() && newBeliefs < maxNewBeliefs) {
            const auto [belief, gap, beliefProbability, depth, path] = queue.top();
            queue.pop();

            // We add the new belief in the history, to avoid adding to the
            // queue the same belief multiple times. We also limit the size of
            // the history to avoid the check taking too much time, we tend to
            // go deeper in the belief tree so it shouldn't be too dangerous.
            if (visitedBeliefs.size() == maxVisitedBeliefs) {
                visitedBeliefs[overwriteCounter] = belief;
                overwriteCounter = (overwriteCounter + 1) % maxVisitedBeliefs;
            } else {
                visitedBeliefs.push_back(belief);
            }

            // We find the best action for this belief with respect to both the
            // upperBound and the lowerBound.
            //
            // If the found actions improve on the bounds, then we'll add this
            // belief to the list.
            const auto [ubAction, ubActionValue] = bestPromisingAction(pomdp, belief, ubQ, ubV);
            const auto [lbAction, lbActionValue] = bestConservativeAction(pomdp, belief, lbVList);

            /***********************
             **     UPPER GAP     **
             ***********************/

            const auto validForUb = [&newUbBeliefs, &ubV](const Belief & b) {
                // We don't consider corners
                for (auto i = 0; i < b.size(); ++i)
                    if (checkEqualSmall(b[i], 0.0) || checkEqualSmall(b[i], 1.0))
                        return false;

                // We also want to check whether we have already added this belief somewhere else.
                const auto check = [&b](const Belief & bb){ return checkEqualProbability(b, bb); };

                auto it = std::find_if(std::begin(newUbBeliefs), std::end(newUbBeliefs), check);
                if (it != std::end(newUbBeliefs))
                    return false;

                it = std::find_if(std::begin(ubV.first), std::end(ubV.first), check);
                if (it != std::end(ubV.first))
                    return false;
                return true;
            };

            if (validForUb(belief)) {
                const auto currentUpperBound = UB(belief, ubQ, ubV);
                if (checkDifferentGeneral(ubActionValue, currentUpperBound) && ubActionValue < currentUpperBound) {
                    newUbBeliefs.push_back(belief);
                    newUbValues.push_back(ubActionValue);

                    // Find all beliefs that brought us here we didn't already have.
                    // Again, we don't consider corners.
                    for (const auto & p : path) {
                        if (validForUb(p)) {
                            newUbBeliefs.push_back(p);
                            newUbValues.push_back(UB(p, ubQ, ubV));
                        }
                    }
                    // Note we only count a single belief even if we added more via
                    // the path (as per original code).
                    ++newBeliefs;
                }
            }

            /***********************
             **     LOWER GAP     **
             ***********************/

            // For the lower gap we don't care about corners (as per original
            // code). We still check on the lower bound lists though.
            const auto validForLb = [&newLbBeliefs, &lbBeliefs](const Belief & b) {
                const auto check = [&b](const Belief & bb){ return checkEqualProbability(b, bb); };

                auto it = std::find_if(std::begin(newLbBeliefs), std::end(newLbBeliefs), check);
                if (it != std::end(newLbBeliefs))
                    return false;

                it = std::find_if(std::begin(lbBeliefs), std::end(lbBeliefs), check);
                if (it != std::end(lbBeliefs))
                    return false;
                return true;
            };

            if (validForLb) {
                double currentLowerBound;
                findBestAtBelief(belief, std::begin(lbVList), std::end(lbVList), &currentLowerBound);
                if (checkDifferentGeneral(lbActionValue, currentLowerBound) && lbActionValue > currentLowerBound) {
                    // We add the new belief, and the same is done for all
                    // beliefs that led us to this one (if they're valid -
                    // i.e., we didn't have them already).
                    newLbBeliefs.push_back(belief);

                    for (const auto & p : path) {
                        if (validForLb(p)) {
                            newLbBeliefs.push_back(p);
                        }
                    }
                    // Note we only count a single belief even if we added more via
                    // the path (as per original code).
                    ++newBeliefs;
                }
            }

            /***********************
             **  QUEUE EXPANSION  **
             ***********************/

            // Avoid it if we're already done anyway.
            if (newBeliefs >= maxNewBeliefs)
                break;

            auto newPath = path;
            newPath.push_back(belief);

            // For each new possible belief, we look if we've already visited
            // it. If not, we compute the gap at that point, and we add it to
            // the queue.
            const Belief intermediateBelief = updateBeliefPartial(pomdp, belief, ubAction);
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                Belief nextBelief = updateBeliefUnnormalized(pomdp, intermediateBelief, ubAction, o);

                const auto nextBeliefProbability = nextBelief.sum();
                if (checkEqualSmall(nextBeliefProbability, 0.0)) continue;
                nextBelief /= nextBeliefProbability;

                const auto check = [&nextBelief](const Belief & bb){ return checkEqualProbability(nextBelief, bb); };
                auto it = std::find_if(std::begin(visitedBeliefs), std::end(visitedBeliefs), check);
                if (it != std::end(visitedBeliefs)) continue;

                const double ubValue = UB(nextBelief, ubQ, ubV);
                double lbValue;
                findBestAtBelief(nextBelief, std::begin(lbVList), std::end(lbVList), &lbValue);

                if ((ubValue - lbValue) * std::pow(pomdp.getDiscount(), depth) > epsilon_ * 20) {
                    const auto nextBeliefOverallProbability = nextBeliefProbability * beliefProbability * pomdp.getDiscount();
                    const auto nextBeliefGap = nextBeliefOverallProbability * (ubValue - lbValue);
                    queue.emplace(
                            std::move(nextBelief),
                            nextBeliefGap,
                            nextBeliefOverallProbability,
                            depth+1,
                            newPath
                    ); // To add: ubValue + lbValue; double paths
                }
            }
        }
        return std::make_tuple(std::move(newLbBeliefs), std::move(newUbBeliefs), std::move(newUbValues));
    }
}

#endif

