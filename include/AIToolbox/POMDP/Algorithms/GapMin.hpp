#ifndef AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE
#define AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE

#include <algorithm>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/Impl/Logging.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>
#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

#include <AIToolbox/LP.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the GapMin algorithm.
     */
    class GapMin {
        public:
            using IntermediatePOMDP = Model<MDP::Model>;
            /**
             * @brief Basic constructor.
             */
            GapMin(double epsilon);

            /**
             * @brief This function solves a POMDP::Model approximately.
             */
            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<double, double, VList, MDP::QFunction> operator()(const M & model, const Belief & initialBelief);

        private:

            using UbVType = std::pair<std::vector<Belief>, std::vector<double>>;

            // Queue sorted by gap.         belief,    gap,   prob,    depth,       path
            using QueueElement = std::tuple<Belief, double, double, unsigned, std::vector<Belief>>;

            class GapTupleLess {
                public:
                    bool operator() (const QueueElement& arg1, const QueueElement& arg2) const;
            };

            using QueueType = boost::heap::fibonacci_heap<QueueElement, boost::heap::compare<GapTupleLess>>;

            void cleanUp(const MDP::QFunction & ubQ, UbVType * ubV, Matrix2D * fibQ);
            std::tuple<double, Vector> UB(const Belief & belief, const MDP::QFunction & ubQ, const UbVType & ubV);

            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<IntermediatePOMDP, Matrix4D> makeNewPomdp(const M& model, const MDP::QFunction &, const UbVType &);

            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<size_t, double> bestPromisingAction(const M & pomdp, const Belief & belief, const MDP::QFunction & ubQ, const GapMin::UbVType & ubV);

            template <typename M, typename = std::enable_if_t<is_model<M>::value>>
            std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> selectReachableBeliefs(
                const M & model,
                const Belief &,
                const VList &,
                const std::vector<Belief> &,
                const MDP::QFunction &, const UbVType &
            );

            double epsilon_;
    };

    GapMin::GapMin(double epsilon) : epsilon_(epsilon) {}

    bool GapMin::GapTupleLess::operator() (const QueueElement& arg1, const QueueElement& arg2) const
    {
        return std::get<1>(arg1) < std::get<1>(arg2);
    }

    template <typename M, typename>
    std::tuple<double, double, VList, MDP::QFunction> GapMin::operator()(const M & pomdp, const Belief & initialBelief) {
        constexpr unsigned infiniteHorizon = 1000000;

        // Helper methods
        BlindStrategies bs(infiniteHorizon, epsilon_);
        FastInformedBound fib(infiniteHorizon, epsilon_);
        PBVI pbvi(0, infiniteHorizon, epsilon_);

        // Here we use the BlindStrategies in order to obtain a very simple
        // initial lower bound.
        VList lbVList;
        std::tie(std::ignore, lbVList) = bs(pomdp, true);
        {
            const auto unwrap = +[](VEntry & ve) -> MDP::Values & {return std::get<VALUES>(ve);};
            const auto rbegin = boost::make_transform_iterator(std::begin(lbVList), unwrap);
            const auto rend   = boost::make_transform_iterator(std::end  (lbVList), unwrap);

            lbVList.erase(extractDominated(pomdp.getS(), rbegin, rend).base(), std::end(lbVList));
        }
        auto lbBeliefs = std::vector<Belief>{initialBelief};

        // The same we do here with FIB for the input POMDP.
        MDP::QFunction ubQ;
        std::tie(std::ignore, ubQ) = fib(pomdp);
        AI_LOGGER(AI_SEVERITY_DEBUG, "Initial QFunction:\n" << ubQ);
        AI_LOGGER(AI_SEVERITY_DEBUG, "Initial belief:\n" << initialBelief);

        // At the same time, we start initializing fibQ, which will be our
        // pseudo-alphaVector storage for our belief-POMDPs which we'll create
        // later.
        //
        // The basic idea is to create a new POMDP where each state is a belief
        // of the input POMDP. This allows us to obtain better upper bounds,
        // and project them to our input POMDP.
        auto fibQ = Matrix2D(pomdp.getS()+1, pomdp.getA());
        fibQ.topLeftCorner(pomdp.getS(), pomdp.getA()).noalias() = ubQ;
        fibQ.row(pomdp.getS()).noalias() = initialBelief.transpose() * ubQ;

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
        AI_LOGGER(AI_SEVERITY_WARNING, "Initial bounds: " << lb << ", " << ub);
        while (var >= epsilon_) {
            AI_LOGGER(AI_SEVERITY_ERROR, "Beginning loop...");
            // Now we find beliefs for both lower and upper bound where we
            // think we can improve. For the ub beliefs we also return their
            // values, since we need them to improve the ub.
            auto [newLbBeliefs, newUbBeliefs, newUbVals] = selectReachableBeliefs(pomdp, initialBelief, lbVList, lbBeliefs, ubQ, ubV);
            const auto newLbBeliefsSize = newLbBeliefs.size();
            const auto newUbBeliefsSize = newUbBeliefs.size();

            AI_LOGGER(AI_SEVERITY_ERROR, "Checking lower bound updates...");
            if (newLbBeliefsSize > 0) {
                AI_LOGGER(AI_SEVERITY_WARNING, "Adding " << newLbBeliefsSize << " new beliefs...");
                for (const auto & b : newLbBeliefs)
                    AI_LOGGER(AI_SEVERITY_DEBUG, "- Belief: " << b.transpose());
                // If we found something interesting for the lower bound, we
                // add it to the beliefs we already had, and we rerun PBVI.
                std::move(std::begin(newLbBeliefs), std::end(newLbBeliefs), std::back_inserter(lbBeliefs));

                // Then we remove all beliefs which don't actively support any
                // alphaVectors.
                AI_LOGGER(AI_SEVERITY_WARNING, "Running PBVI...");
                lbVList = std::move(std::get<1>(pbvi(pomdp, lbBeliefs, ValueFunction{std::move(lbVList)})).back());
                AI_LOGGER(AI_SEVERITY_DEBUG, "Pruning non-useful beliefs, from " << lbBeliefs.size() << " beliefs...");
                lbBeliefs.erase(extractUsefulBeliefs(
                    std::begin(lbBeliefs), std::end(lbBeliefs),
                    std::begin(lbVList), std::end(lbVList)),
                    std::end(lbBeliefs)
                );
                AI_LOGGER(AI_SEVERITY_DEBUG, "... to " << lbBeliefs.size() << " beliefs.");

                // And we recompute the lower bound.
                findBestAtBelief(initialBelief, std::begin(lbVList), std::end(lbVList), &lb);
                AI_LOGGER(AI_SEVERITY_WARNING, "Updated lower bound to " << lb);
            }

            AI_LOGGER(AI_SEVERITY_ERROR, "Checking upper bound updates...");
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

                AI_LOGGER(AI_SEVERITY_WARNING, "Building new POMDP...");
                // We create a new POMDP where each state is a belief.
                auto [newPOMDP, newPOMDPSOSA] = makeNewPomdp(pomdp, ubQ, ubV);
                AI_LOGGER(AI_SEVERITY_WARNING, "Running FIB...");
                // And we approximate its upper bound.
                std::tie(std::ignore, fibQ) = fib(newPOMDP, newPOMDPSOSA, std::move(fibQ));

                // We extract from the found upper bound the part for the
                // states of the input POMDP, and we copy them to our
                // upperBound alphavectors. We additionally update the values
                // for all ub beliefs.
                AI_LOGGER(AI_SEVERITY_WARNING, "Updating UbQ...");
                ubQ.noalias() = fibQ.block(0, 0, pomdp.getS(), pomdp.getA());
                for (size_t i = 0; i < ubV.second.size(); ++i)
                    ubV.second[i] = fibQ.row(pomdp.getS() + i).maxCoeff();

                // Finally, we remove some unused stuff, and we recompute the upperbound.
                AI_LOGGER(AI_SEVERITY_WARNING, "Cleanup...");
                cleanUp(ubQ, &ubV, &fibQ);
                AI_LOGGER(AI_SEVERITY_WARNING, "Recomputing upper bound...");
                std::tie(ub, std::ignore) = UB(initialBelief, ubQ, ubV);

                AI_LOGGER(AI_SEVERITY_WARNING, "Updated upper bound to " << ub);
            }

            // Update the difference between upper and lower bound so we can
            // return it/use it to stop the loop.
            var = ub - lb;
            AI_LOGGER(AI_SEVERITY_ERROR, "Updated bounds to " << lb << ", " << ub);

            // Stop if we didn't find anything new.
            if (newLbBeliefsSize + newUbBeliefsSize == 0)
                break;
        }
        return std::make_tuple(lb, ub, lbVList, ubQ);
    }

    void GapMin::cleanUp(const MDP::QFunction & ubQ, UbVType * ubVp, Matrix2D * fibQp) {
        assert(ubVp);
        assert(fibQp);

        UbVType & ubV = *ubVp;
        Matrix2D & fibQ = *fibQp;

        std::vector<size_t> toRemove;
        size_t i;

        // For each belief, we try to compute its upper bound with the others.
        // If it doesn't change, it means that we don't really need it, and
        // thus we remove it.
        i = 0;
        while (i < ubV.first.size() && ubV.first.size() > 1) {
            std::swap(ubV.first[i], ubV.first.back());
            std::swap(ubV.second[i], ubV.second.back());

            auto belief = std::move(ubV.first.back());
            auto value = ubV.second.back();

            ubV.first.pop_back();
            ubV.second.pop_back();

            const auto [v, dist] = UB(belief, ubQ, ubV);
            (void)dist;

            if (std::fabs(v - value) < epsilon_) {
                // We keep note of the "real" indeces of the removed beliefs.
                toRemove.push_back(i + toRemove.size());
            } else {
                // Unpop and unswap, since we need to keep the order consistent
                // (fibQ depends on it). This could be done with a couple less
                // moves but like this it's more clear.
                ubV.first.emplace_back(std::move(belief));
                ubV.second.emplace_back(value);

                std::swap(ubV.first[i], ubV.first.back());
                std::swap(ubV.second[i], ubV.second.back());

                ++i;
            }
        }
        // If all beliefs are useful, we're done.
        if (toRemove.size() == 0) return;

        // Here we do a bit of fancy dance in order to do as little operations
        // as possible to remove the unneeded rows from fibQ. Additionally, we
        // try to also do as few block operations as possible, in the hope that
        // Eigen gets to optimize those and thus speeds us up.
        //
        // We need to allocate another matrix anyway since if we copied from
        // fibQ to fibQ we'd have to make an allocation per copy anyway. So we
        // do only one and we're done.
        //
        // The idea is simply copying one block at a time towards the beginning.
        const size_t newRows = fibQ.rows() - toRemove.size();
        const size_t cols = fibQ.cols();
        Matrix2D newFibQ(newRows, cols);

        size_t beginSource = 0;
        size_t beginTarget = 0;
        size_t toCopy;

        i = 0;
        while (i < toRemove.size()) {
            while (i < toRemove.size() && toRemove[i] == beginSource) ++i, ++beginSource;

            if (i == toRemove.size()) {
                if (beginSource > cols) break;
                toCopy = cols - beginSource;
            } else {
                toCopy = toRemove[i] - beginSource;
            }

            newFibQ.block(beginTarget, 0, toCopy, cols) = fibQ.block(beginSource, 0, toCopy, cols);

            beginTarget += toCopy;
            beginSource += toCopy;
        }
        // And we give it back
        fibQ = std::move(newFibQ);
    }

    template <typename M, typename>
    std::tuple<GapMin::IntermediatePOMDP, Matrix4D> GapMin::makeNewPomdp(const M& model, const MDP::QFunction & ubQ, const UbVType & ubV) {
        size_t S = model.getS() + ubV.first.size();

        // First we build the new reward function. For normal states, this is
        // the same as the old one. For all additional states (beliefs), we
        // simply take their expected reward with respect to the original
        // reward function.
        Matrix2D R(S, model.getA());
        const auto & ir = [&]{
            if constexpr (MDP::is_model_eigen<M>::value) return model.getRewardFunction();
            else return computeImmediateRewards(model);
        }();

        R.topLeftCorner(model.getS(), model.getA()).noalias() = ir;
        for (size_t b = 0; b < ubV.first.size(); ++b)
            R.row(model.getS()+b) = ubV.first[b].transpose() * ir;

        // Now we create the SOSA matrix for this new POMDP. For each pair of
        // action/observation, and for each belief we have (thus state), we
        // compute the probability of going to any other belief.
        //
        // This is done through the UB function, although I must admit I don't
        // fully understand the math behind of why it works.
        Belief helper(model.getS()), corner(model.getS());
        corner.fill(0.0);

        Matrix4D sosa( boost::extents[model.getA()][model.getO()] );
        Matrix2D m(S, S);
        const auto updateMatrix = [&](const Belief & b, size_t a, size_t o, size_t index) {
            updateBeliefUnnormalized(model, b, a, o, &helper);
            auto sum = helper.sum();
            if (checkEqualSmall(sum, 0.0)) {
                m.row(index).fill(0.0);
            } else {
                Vector dist;
                std::tie(std::ignore, dist) = UB(helper/sum, ubQ, ubV);
                m.row(index).noalias() = dist * sum;
            }
        };

        for (size_t a = 0; a < model.getA(); ++a) {
            for (size_t o = 0; o < model.getO(); ++o) {
                for (size_t s = 0; s < model.getS(); ++s) {
                    corner[s] = 1.0;
                    updateMatrix(corner, a, o, s);
                    corner[s] = 0.0;
                }

                for (size_t b = 0; b < ubV.first.size(); ++b)
                    updateMatrix(ubV.first[b], a, o, model.getS() + b);

                // After updating all rows of the matrix, we put it inside the
                // SOSA table.
                sosa[a][o] = m;
            }
        }

        // Finally we return a POMDP with no transition nor observation
        // function, since those are contained in the SOSA matrix.
        //
        // We do however include the new reward function that contains rewards
        // for each new "state"/belief.
        return std::make_tuple(
            IntermediatePOMDP(
                NO_CHECK, model.getO(), Matrix3D(),
                NO_CHECK, S, model.getA(), Matrix3D(), std::move(R), model.getDiscount()
            ),
            std::move(sosa)
        );
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
    template <typename M, typename = std::enable_if_t<is_model<M>::value>>
    std::tuple<size_t, double> bestConservativeAction(const M & pomdp, const Belief & initialBelief, const VList & lbVList) {
        MDP::QFunction ir = [&]{
            if constexpr (MDP::is_model_eigen<M>::value)
                return pomdp.getRewardFunction();
            else
                return computeImmediateRewards(pomdp);
        }();

        Vector bpAlpha(pomdp.getS());
        for (size_t a = 0; a < pomdp.getA(); ++a) {
            const Belief intermediateBelief = updateBeliefPartial(pomdp, initialBelief, a);

            bpAlpha.fill(0.0);

            for (size_t o = 0; o < pomdp.getO(); ++o) {
                Belief nextBelief = updateBeliefPartialUnnormalized(pomdp, initialBelief, a, o);

                const auto nextBeliefProbability = nextBelief.sum();
                if (checkEqualSmall(nextBeliefProbability, 0.0)) continue;
                // Now normalized
                nextBelief /= nextBeliefProbability;

                auto it = findBestAtBelief(nextBelief, std::begin(lbVList), std::end(lbVList));

                bpAlpha += pomdp.getTransitionFunction(a).col(o).cwiseProduct(std::get<VALUES>(*it));
            }
            ir.col(a) += pomdp.getDiscount() * pomdp.getTransitionFunction(a) * bpAlpha;
        }

        size_t id;
        double v = (initialBelief.transpose() * ir).maxCoeff(&id);

        return std::make_tuple(id, v);
    }

    template <typename M, typename>
    std::tuple<size_t, double> GapMin::bestPromisingAction(const M & pomdp, const Belief & belief, const MDP::QFunction & ubQ, const GapMin::UbVType & ubV) {
        Vector qvals = belief.transpose() * [&]{
            if constexpr (MDP::is_model_eigen<M>::value)
                return pomdp.getRewardFunction();
            else
                return computeImmediateRewards(pomdp);
        }();

        double val;
        for (size_t a = 0; a < pomdp.getA(); ++a) {
            const Belief intermediateBelief = updateBeliefPartial(pomdp, belief, a);
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                Belief nextBelief = updateBeliefUnnormalized(pomdp, intermediateBelief, a, o);

                const auto sum = nextBelief.sum();
                if (checkEqualSmall(sum, 0.0)) continue;
                nextBelief /= sum;

                AI_LOGGER(AI_SEVERITY_DEBUG, "");
                std::tie(val, std::ignore) = UB(nextBelief, ubQ, ubV);
                qvals[a] += pomdp.getDiscount() * sum * val;
            }
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "");
        size_t bestAction;
        double bestValue = qvals.maxCoeff(&bestAction);

        return std::make_tuple(bestAction, bestValue);
    }

    template <typename M, typename>
    std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> GapMin::selectReachableBeliefs(
            const M & pomdp, const Belief & initialBelief, const VList & lbVList,
            const std::vector<Belief> & lbBeliefs, const MDP::QFunction & ubQ, const UbVType & ubV
        )
    {
        AI_LOGGER(AI_SEVERITY_WARNING, "Selecting reachable beliefs...");
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
        queue.emplace(QueueElement(initialBelief, 0.0, 1.0, 1, {}));

        while (!queue.empty() && newBeliefs < maxNewBeliefs) {
            const auto [belief, gap, beliefProbability, depth, path] = queue.top();
            AI_LOGGER(AI_SEVERITY_INFO, "### QUEUE EXTRACTION: Now examining belief: " << belief.transpose());
            (void)gap; // ignore gap variable
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
            AI_LOGGER(AI_SEVERITY_INFO, "Best promising UB action: " << ubAction << " with value " << ubActionValue);
            AI_LOGGER(AI_SEVERITY_INFO, "Best conservative LB action: " << lbAction << " with value " << lbActionValue);

            (void)lbAction; // ignore lbAction

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
                {
                    auto it = std::find_if(std::begin(newUbBeliefs), std::end(newUbBeliefs), check);
                    if (it != std::end(newUbBeliefs))
                        return false;
                }
                {
                    auto it = std::find_if(std::begin(ubV.first), std::end(ubV.first), check);
                    if (it != std::end(ubV.first))
                        return false;
                }
                return true;
            };

            if (validForUb(belief)) {
                AI_LOGGER(AI_SEVERITY_INFO, "UB: belief " << belief.transpose() << " is valid for UB.");
                const double currentUpperBound = std::get<0>(UB(belief, ubQ, ubV));
                AI_LOGGER(AI_SEVERITY_INFO, "UB: This belief currently has a value of " << currentUpperBound << " vs the found " << ubActionValue);
                if (checkDifferentGeneral(ubActionValue, currentUpperBound) && ubActionValue < currentUpperBound) {
                    AI_LOGGER(AI_SEVERITY_INFO, "UB: Adding this belief to the UB with value " << ubActionValue);
                    newUbBeliefs.push_back(belief);
                    newUbValues.push_back(ubActionValue);

                    // Find all beliefs that brought us here we didn't already have.
                    // Again, we don't consider corners.
                    for (const auto & p : path) {
                        if (validForUb(p)) {
                            newUbBeliefs.push_back(p);
                            newUbValues.push_back(std::get<0>(UB(p, ubQ, ubV)));
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
                {
                    auto it = std::find_if(std::begin(newLbBeliefs), std::end(newLbBeliefs), check);
                    if (it != std::end(newLbBeliefs))
                        return false;
                }
                {
                    auto it = std::find_if(std::begin(lbBeliefs), std::end(lbBeliefs), check);
                    if (it != std::end(lbBeliefs))
                        return false;
                }
                return true;
            };

            if (validForLb(belief)) {
                AI_LOGGER(AI_SEVERITY_INFO, "LB: belief " << belief.transpose() << " is valid for LB.");
                double currentLowerBound;
                findBestAtBelief(belief, std::begin(lbVList), std::end(lbVList), &currentLowerBound);
                AI_LOGGER(AI_SEVERITY_INFO, "LB: This belief currently has a value of " << currentLowerBound << " vs the found " << lbActionValue);
                if (checkDifferentGeneral(lbActionValue, currentLowerBound) && lbActionValue > currentLowerBound) {
                    AI_LOGGER(AI_SEVERITY_INFO, "LB: Adding this belief to the LB with value " << lbActionValue);
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

            AI_LOGGER(AI_SEVERITY_INFO, "Expanding queue...");

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

                const double ubValue = std::get<0>(UB(nextBelief, ubQ, ubV));
                double lbValue;
                findBestAtBelief(nextBelief, std::begin(lbVList), std::end(lbVList), &lbValue);

                if ((ubValue - lbValue) * std::pow(pomdp.getDiscount(), depth) > epsilon_ * 20) {
                    AI_LOGGER(AI_SEVERITY_INFO, "Adding belief " << nextBelief.transpose() << ", which we don't already have...");
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

    std::tuple<double, Vector> GapMin::UB(const Belief & belief, const MDP::QFunction & ubQ, const UbVType & ubV) {
        AI_LOGGER(AI_SEVERITY_DEBUG, "Starting upper bound LP for belief " << belief.transpose() << "...");
        // Here we find all beliefs that have the same "zeroes" as the input one.
        // This is done to reduce the amount of work the LP has to do.
        std::vector<size_t> zeroStates;
        std::vector<size_t> nonZeroStates;
        for (size_t s = 0; s < static_cast<size_t>(belief.size()); ++s) {
            if (checkEqualSmall(belief[s], 0.0))
                zeroStates.push_back(s);
            else
                nonZeroStates.push_back(s);
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "Non-zero states for belief are " << nonZeroStates.size() << "...");

        std::vector<size_t> compatibleBeliefs;
        if (zeroStates.size() == 0) {
            // If no zero states, we match all beliefs.
            compatibleBeliefs.resize(ubV.first.size());
            std::iota(std::begin(compatibleBeliefs), std::end(compatibleBeliefs), 0);
        } else {
            for (size_t i = 0; i < ubV.first.size(); ++i) {
                bool add = true;
                for (const auto s : zeroStates) {
                    if (checkDifferentSmall(ubV.first[i][s], 0.0)) {
                        add = false;
                        break;
                    }
                }
                if (add) compatibleBeliefs.push_back(i);
            }
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "Found " << compatibleBeliefs.size() << " compatible beliefs.");

        // If there's no other belief on the same plane as this one, the V can't
        // help us with the bound. So we just use the Q, and we copy its values in the
        // corners of the belief.
        if (compatibleBeliefs.size() == 0) {
            AI_LOGGER(AI_SEVERITY_DEBUG, "Returning simple product between belief and input Q.");
            Vector retval(belief.size() + ubV.first.size());

            retval.head(belief.size()).noalias() = belief;
            retval.tail(ubV.first.size() - belief.size()).fill(0.0);

            return std::make_tuple((belief.transpose() * ubQ).maxCoeff(), std::move(retval));
        }

        Vector cornerVals = ubQ.rowwise().maxCoeff();
        AI_LOGGER(AI_SEVERITY_DEBUG, "Values at the corners are: " << cornerVals.transpose());

        double unscaledValue;
        Vector result;

        // If there's only a single compatible belief, we don't really need to run
        // an LP.
        if (compatibleBeliefs.size() == 1) {
            AI_LOGGER(AI_SEVERITY_DEBUG, "Only one belief is compatible, we can bypass the LP...");

            const auto & compBelief = ubV.first[compatibleBeliefs[0]];
            AI_LOGGER(AI_SEVERITY_DEBUG, "The compatible belief is " << compBelief.transpose() << " with value " << ubV.second[compatibleBeliefs[0]]);

            result.resize(1);
            result[0] = (belief.cwiseQuotient(compBelief)).minCoeff();

            unscaledValue = result[0] * (ubV.second[compatibleBeliefs[0]] - compBelief.transpose() * cornerVals);
        } else {
            /*
             * Here we run the LP.
             *
             * In order to obtain the linear approximation for the upper bound of
             * the input belief, given that we already know the values for a set of
             * beliefs, we need to solve an LP in the form:
             *
             * c[0] * b[0][0] + c[1] * b[1][0] + ...                = bin[0]
             * c[0] * b[0][1] + c[1] * b[1][1] + ...                = bin[1]
             * c[0] * b[0][2] + c[1] * b[1][2] + ...                = bin[2]
             * ...
             * c[0] * v[0]    + c[1] * v[1]    + ... - K            = 0
             *
             * And we minimize K to get:
             *
             * argmin(c) = sum( c * v ) = K
             *
             * This way K will be the minimum upper bound possible for the input
             * belief (bin), found by interpolating all other known beliefs. At the
             * same time we apply the linear approximation by enforcing
             *
             * sum( c * b ) = bin
             *
             * We also set each c to be >= 0.
             *
             * OPTIMIZATIONS:
             *
             * Once we have defined the problem, we can apply a series of
             * optimizations to reduce the size of the LP to be solved. These were
             * taken from the MATLAB code published for the GapMin algorithm. Written
             * interpretation below is mine.
             *
             * - Nonzero & Compatible Beliefs
             *
             * If the input belief is restricted to a subset of dimensions in the
             * VFunction (meaning some of its values are zero), and we have beliefs
             * in that exact same subset, we can just use those in order to determine
             * the upper bound of the input. This is true since additional dimensions
             * won't affect the ValueFunction in the particular subspace the input
             * belief is in. All other beliefs are discarded. Note that we'll need
             * to fill in zeroes for the coefficients of the discarded beliefs after
             * we are done.
             *
             * - Removal of Corner Values
             *
             * Ideally, one would want the corner beliefs/values to be included in
             * the list of beliefs to use for interpolation, since they are needed.
             * However, all other belief values can simply be scaled down as if the
             * corner values were zero, and the resulting solution would not change.
             * The only thing is that the values obtained for the target function
             * would need to be scaled back before being returned by the LP.
             *
             * The other thing is that removing the corner values from the
             * beliefs also means removing the coefficients for them in the LP.
             * This is good as the LP is simpler, but pretty much makes it
             * infeasible. So what we do is change the constraints, and instead
             * of making them equal, we make them less or equal.
             *
             * In the end we're going to fix all missing numbers anyway by
             * filling the corners with the needed numbers.
             */
            AI_LOGGER(AI_SEVERITY_DEBUG, "Setting up the LP...");

            // We're going to have one column per compatible belief (plus one, but
            // that's implied).
            LP lp(compatibleBeliefs.size() + 1);
            lp.resize(nonZeroStates.size() + 1); // One row per state, plus the K constraint.
            size_t i;

            // Goal: minimize K.
            lp.setObjective(compatibleBeliefs.size(), false);

            // IMPORTANT: K is unbounded, since the value function may be negative.
            lp.setUnbounded(compatibleBeliefs.size());

            // By default we don't have K, only at the end.
            lp.row[compatibleBeliefs.size()] = +0.0;

            // So each row contains the same-index element from all the compatible
            // beliefs, and they should sum up to that same element in the input belief.
            for (const auto s : nonZeroStates) {
                i = 0;
                for (const auto b : compatibleBeliefs)
                    lp.row[i++] = ubV.first[b][s];
                lp.pushRow(LP::Constraint::LessEqual, belief[s]);
            }

            // Finally we setup the last row.
            i = 0;
            for (const auto b : compatibleBeliefs) {
                double val = ubV.second[b];
                for (const auto s : nonZeroStates)
                    val -= ubV.first[b][s] * cornerVals[s];
                lp.row[i++] = val;
            }
            lp.row[i] = -1.0;
            lp.pushRow(LP::Constraint::Equal, 0.0);

            // Now solve
            AI_LOGGER(AI_SEVERITY_DEBUG, "Launching solve process...");
            auto tmp = lp.solve(compatibleBeliefs.size(), &unscaledValue);
            if (!tmp)
                throw std::runtime_error("GapMin UB process failed!");
            result = *tmp;
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "Unscaled solution value is " << unscaledValue << "...");

        // We scale back the value as if we had considered the corners.
        double ubValue = unscaledValue + belief.transpose() * cornerVals;

        Vector retval(belief.size() + ubV.first.size());
        retval.fill(0.0);

        // And we fix the coefficients in order to actually make the equalities
        // hold.
        for (const auto s : nonZeroStates) {
            double sum = 0.0;
            for (size_t i = 0; i < compatibleBeliefs.size(); ++i)
                sum += ubV.first[compatibleBeliefs[i]][s] * result[i];
            retval[s] = belief[s] - sum;
            // Remove infinitesimal values
            if (checkEqualSmall(retval[s], 0.0)) retval[s] = 0.0;
        }
        retval.tail(compatibleBeliefs.size()) = result;

        AI_LOGGER(AI_SEVERITY_DEBUG, "Result is: " << retval.transpose());
        AI_LOGGER(AI_SEVERITY_DEBUG, "Value is: " << ubValue);
        return std::make_tuple(ubValue, std::move(retval));
    }
}

#endif

