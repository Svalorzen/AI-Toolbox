#ifndef AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE
#define AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE

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
            std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> selectReachableBeliefs(const M & model, const VList &, const MDP::QFunction &, const UbVType &);

            double epsilon_;
    };

    template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
    std::tuple<double, ValueFunction> GapMin::operator()(const M & pomdp, const Belief & initialBelief) {
        constexpr unsigned infiniteHorizon = 1000000;

        // Helper methods
        BlindStrategies bs(infiniteHorizon, epsilon_);
        FastInformedBound fib(infiniteHorizon, epsilon_);
        PBVI pbvi(0, infiniteHorizon, epsilon_);

        auto lbVList = bs(pomdp, true);
        {
            const auto unwrap = +[](VEntry & ve) -> MDP::Values & {return std::get<VALUES>(ve);};
            const auto rbegin = boost::make_transform_iterator(std::begin(lbVList), unwrap);
            const auto rend   = boost::make_transform_iterator(std::end  (lbVList), unwrap);

            lbVList.erase(extractDominated(pomdp.getS(), rbegin, rend).base(), std::end(lbVList));
        }
        auto lbBeliefs = std::vector<Belief>{initialBelief};

        auto ubQ = fib(pomdp);

        auto fibQ = Matrix2D(pomdp.getS()+1, pomdp.getA());
        fibQ.block(0, 0, pomdp.getS(), pomdp.getA()).noalias() = ubQ;
        fibQ.row(pomdp.getS()).noalias() = ubQ * initialBelief;

        UbVType ubV = {
            {initialBelief}, {fibQ.row(pomdp.getS()).maxCoeff()}
        };

        // Init begin ub/lb
        double lb;
        findBestAtBelief(initialBelief, std::begin(lbVList), std::end(lbVList), &lb);
        double ub = ubV.second[0];

        auto var = ub - lb;
        while (var >= epsilon_) {
            auto [newLbBeliefs, newUbBeliefs, newUbVals] = selectReachableBeliefs(pomdp, lbVList, ubQ, ubV);
            const auto newLbBeliefsSize = newLbBeliefs.size();
            const auto newUbBeliefsSize = newUbBeliefs.size();

            if (newLbBeliefsSize > 0) {
                std::move(std::begin(newLbBeliefs), std::end(newLbBeliefs), std::back_inserter(lbBeliefs));

                lbVList = std::move(pbvi(lbBeliefs, VFunction(lbVList)).back());
                lbBeliefs.erase(extractUsefulBeliefs(
                    std::begin(lbBeliefs), std::end(lbBeliefs),
                    std::begin(lbVList), std::end(lbVList)),
                    std::end(lbBeliefs)
                );

                findBestAtBelief(initialBelief, std::begin(lbVList), std::end(lbVList), &lb);
            }

            if (newUbBeliefsSize > 0) {
                const auto prevRows = pomdp.getS() + ubV.first.size();
                fibQ.conservativeResize(prevRows + newUbBeliefsSize, Eigen::NoChange);

                for (size_t i = 0; i < newUbBeliefs.size(); ++i) {
                    ubV.first.emplace_back(std::move(newUbBeliefs[i]));
                    ubV.second.emplace_back(newUbVals[i]);
                    fibQ.row(prevRows + i).fill(newUbVals[i]);
                }

                auto [newPOMDP, newPOMDPSOSA] = makeNewPomdp(pomdp, ubQ, ubV);
                std::tie(std::ignore, fibQ) = fib(newPOMDP, newPOMDPSOSA, std::move(fibQ));

                ubQ.noalias() = fibQ.block(0, 0, pomdp.getS(), pomdp.getA());
                for (size_t i = 0; i < ubV.second.size(); ++i)
                    ubV.second[i] = fibQ.row(pomdp.getS() + i).maxCoeff();

                cleanUp(ubQ, ubV, fibQ);
                ub = upperBound(initialBelief, ubQ, ubV);
            }

            if (newLbBeliefsSize + newUbBeliefsSize == 0)
                break;

            var = ub - lb;
        }
    }

    template <typename M, typename std::enable_if<is_model<M>::value, int>::type>
    std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> GapMin::selectReachableBeliefs(const M & pomdp, const VList &, const MDP::QFunction &, const UbVType &) {
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

        queue.push_back(initialBelief, 0.0, 1.0, 1, {});
        while (!queue.isEmpty() && newBeliefs < maxNewBeliefs) {
            const auto [belief, gap, beliefProbability, depth, path] = queue.top();
            queue.pop();

            // Limit history, we tend to go deeper so it shouldn't be too dangerous.
            if (visitedBeliefs.size() == maxVisitedBeliefs)
                visitedBeliefs[overwriteCounter++] = belief;
            else
                visitedBeliefs.push_back(belief);

            const auto [ubAction, ubActionValue] = bestPromisingAction(pomdp, belief, ubQ, ubV);
            const auto [lbAction, lbActionValue] = bestConservativeAction(pomdp, belief, lbVList);

            /***********************
             **     UPPER GAP     **
             ***********************/

            if (!(belief is in newUbBeliefs or belief is in ubV or belief is in corners)) {
                auto currentUpperBound = UB(belief, ubQ, ubV);
                if (ubActionValue < currentUpperBound) { // FIXME: Possibly eps
                    uniquePath = beliefSetDiff(path,[ubBeliefSet;ub.beliefSet;corners]);
                    values += ubActionValue;
                    for (const auto & p : uniquePath)
                        values += UB(p, ubQ, ubV);
                    ubBeliefSet += belief, std::move(uniquePath);

                    ++newBeliefs;
                }
            }

            /***********************
             **     LOWER GAP     **
             ***********************/

            if (!(belief is in newLbBeliefs or belief is in lbVList)) {
                auto currentLowerBound = lowerBound(belief,lb.alphaVectors);
                if (lbActionValue > currentLowerBound) { // FIXME: possibly eps
                    uniquePath = beliefSetDiff(path,[lbBeliefSet;lb.beliefSet]);

                    lbBeliefSet += belief; std::move(uniquePath);

                    ++newBeliefs;
                }
            }

            /***********************
             **  QUEUE EXPANSION  **
             ***********************/

            const auto intermediateBelief = (belief.transpose() * pomdp.getTransitionFunction(ubAction)).transpose();
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                // Unnormalized here
                Belief nextBelief = intermediateBelief.cwiseProduct(pomdp.getObservationFunction(ubAction).col(o));

                const auto nextBeliefProbability = nextBelief.sum();
                if (checkEqual(sum, 0.0)) continue;
                // Now normalized
                nextBelief /= nextBeliefProbability;
                if (visitedBelief.contains(nextBelief)) continue;

                const auto ubValue = UB(nextBelief, ubQ, ubV);
                const auto lbValue = LB(nextBelief, lbVList);

                if ((ubValue - lbValue) * std::pow(pomdp.getDiscount(), depth) > eps * 20) {
                    const auto nextBeliefOverallProbability = nextBeliefProbability * beliefProbability * pomdp.getDiscount();
                    const auto nextBeliefGap = nextBeliefOverallProbability * (ubValue - lbValue);
                    queue.emplace(
                            std::move(nextBelief),
                            nextBeliefGap,
                            nextBeliefOverallProbability,
                            depth+1,
                            path + belief
                            ); // To add: ubValue + lbValue; double paths
                }
            }
        }

        return std::make_tuple(std::move(newLbBeliefs), std::move(newUbBeliefs), std::move(newUbValues));
    }
}

#endif

