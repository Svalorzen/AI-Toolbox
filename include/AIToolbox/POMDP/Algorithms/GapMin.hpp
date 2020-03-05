#ifndef AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE
#define AI_TOOLBOX_POMDP_GAPMIN_HEADER_FILE

#include <algorithm>

#include <boost/heap/fibonacci_heap.hpp>

#include <AIToolbox/Impl/Logging.hpp>

#include <AIToolbox/Utils/Polytope.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/POMDP/Model.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>
#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>
#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the GapMin algorithm.
     *
     * This method works by repeatedly refining both a lower bound and upper
     * bound for the input POMDP.
     *
     * The lower bound is worked through PBVI.
     *
     * The upper bound is worked through a combination of alphavectors, and a
     * belief-value pair piecewise linear surface.
     *
     * At each iteration, a set of beliefs are found that the algorithm thinks
     * may be useful to reduce the bound.
     *
     * For the lower bound, these beliefs are added to a list, and run through
     * PBVI. Spurious beliefs are then removed.
     *
     * For the upper bound, the beliefs are used to create a temporary belief
     * POMDP, where each belief is a state. This belief is then used as input
     * to the FastInformedBound algorithm, which refines its upper bound.
     *
     * The strong point of the algorithm is that beliefs are searched by gap
     * size, so that the beliefs that are most likely to decrease the gap are
     * looked at first. This results in less overall work to highly reduce the
     * bound.
     *
     * In order to act, the output lower bound should be used (as it's the only
     * one that gives an actual guarantee), but for this just using PBVI may be
     * more useful.
     */
    class GapMin {
        public:
            /**
             * @brief Basic constructor.
             *
             * The input parameters can heavily influence both the time and the
             * strictness of the resulting bound.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * function will throw an std::runtime_error.
             *
             * \sa setInitialTolerance(double)
             * \sa setPrecisionDigits(unsigned)
             *
             * @param initialTolerance The tolerance to compute the initial bounds.
             * @param precisionDigits The number of digits precision to stop the gap searching process.
             */
            GapMin(double initialTolerance, unsigned precisionDigits);

            /**
             * @brief This function sets the initial tolerance used to compute the initial bounds.
             *
             * This value is only used before having an initial bounds
             * approximation. Once that has been established, the tolerance is
             * dependent on the digits of precision parameter.
             *
             * The tolerance parameter must be >= 0.0, otherwise the
             * function will throw an std::runtime_error.
             *
             * \sa setPrecisionDigits(unsigned);
             *
             * @param initialTolerance The new initial tolerance.
             */
            void setInitialTolerance(double initialTolerance);

            /**
             * @brief This function returns the initial tolerance used to compute the initial bounds.
             *
             * @return The currently set initial tolerance.
             */
            double getInitialTolerance() const;

            /**
             * @brief This function sets the digits in precision for the returned solution.
             *
             * Depending on the values for the input model, the precision of
             * the solution is automatically adjusted to the input precision
             * digits.
             *
             * In particular, the return threshold is equal to:
             *
             *     std::pow(10, std::ceil(std::log10(std::max(std::fabs(ub), std::fabs(lb))))-precisionDigits);
             *
             * This is used in two ways:
             *
             * - To check for lower and upper bound convergence. If the bounds
             *   difference is less than the threshold, the GapMin terminates.
             * - To check for gap size converngence. If the gap has not reduced
             *   by more than the threshold during the last iteration, GapMin
             *   terminates.
             *
             * @param digits The number of digits of precision to use to test for convergence.
             */
            void setPrecisionDigits(unsigned digits);

            /**
             * @brief This function returns the currently set digits of precision.
             *
             * \sa setPrecisionDigits(unsigned);
             *
             * @return The currently set digits of precision to use to test for convergence.
             */
            unsigned getPrecisionDigits() const;

            /**
             * @brief This function efficiently computes bounds for the optimal value of the input belief for the input POMDP.
             *
             * @param model The model to compute the gap for.
             * @param initialBelief The belief to compute the gap for.
             *
             * @return The lower and upper gap bounds, the lower bound VList, and the upper bound QFunction.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<double, double, VList, MDP::QFunction> operator()(const M & model, const Belief & initialBelief);

        private:
            using IntermediatePOMDP = Model<MDP::Model>;

            // Queue sorted by gap.         belief,    gap,   prob,    lb,    ub,    depth,       path
            using QueueElement = std::tuple<Belief, double, double, double, double, unsigned, std::vector<Belief>>;

            struct QueueElementLess {
                bool operator() (const QueueElement& arg1, const QueueElement& arg2) const;
            };

            using QueueType = boost::heap::fibonacci_heap<QueueElement, boost::heap::compare<QueueElementLess>>;

            /**
             * @brief This function collects beliefs in order to reduce the gap.
             *
             * This function explores beliefs and sorts them by gap size. It
             * creates two lists, for lower and upper bound, which contain
             * these beliefs.
             *
             * The gap is computed based on the input lower bound VList, and
             * upper bound QFunction and belief list.
             *
             * The beliefs are explored in a sequential fashion from the input
             * belief.
             *
             * @param model The POMDP model to look beliefs for.
             * @param belief The belief to compute from.
             * @param lbV The VList for the lower bound.
             * @param lbBeliefs The beliefs supporting the lower bound.
             * @param ubQ The QFunction containing the upper bound.
             * @param ubV The belief-value pairs for the upper bound.
             *
             * @return Two lists of beliefs, for lower and upper bound respectively, and a list of values for the upper bound beliefs.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> selectReachableBeliefs(
                const M & model,
                const Belief & belief,
                const VList & lbV,
                const std::vector<Belief> & lbBeliefs,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
            );

            /**
             * @brief This function creates a partial POMDP and its SOSA matrix from the input upper bound.
             *
             * Only the reward matrix is computed for the output POMDP, as it's
             * the only part that matters. For the rest, a SOSA matrix is also
             * computed and returned, so that the two can be jointly used with
             * the FastInformedBound method.
             *
             * The output POMDP has an additional state for each belief
             * contained in the ubV. The SOSA matrix in particular is built so
             * that transition/observation probabilities between beliefs follow
             * the upper bound of the input.
             *
             * @param model The POMDP model to look beliefs for.
             * @param ubQ The QFunction containing the upper bound.
             * @param ubV The belief-value pairs for the upper bound.
             *
             * @return A pair with a reward-function only POMDP, and its associated SOSA matrix.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            std::tuple<IntermediatePOMDP, SparseMatrix4D> makeNewPomdp(const M& model, const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV);

            /**
             * @brief This function skims useless beliefs from the ubV.
             *
             * This function also removes the appropriate lines from the fibQ,
             * since each line in it represents one of the beliefs.
             *
             * The beliefs are removed preferentially in the ones that have
             * been added last.
             *
             * Beliefs are removed when they do not contribute to the
             * belief-value piecewise linear surface of the upper bound.
             *
             * @param ubQ The current basic upper bound.
             * @param ubV The current belief-value pairs.
             * @param fibQ The alphavectors associated with the ubV.
             */
            void cleanUp(const MDP::QFunction & ubQ, UpperBoundValueFunction * ubV, Matrix2D * fibQ);

            Matrix2D immediateRewards_;
            double tolerance_;
            double initialTolerance_;
            unsigned precisionDigits_;
    };

    template <typename M, typename>
    std::tuple<double, double, VList, MDP::QFunction> GapMin::operator()(const M & pomdp, const Belief & initialBelief) {
        constexpr unsigned infiniteHorizon = 1000000;

        // Cache immediate rewards if we can't read the reward function directly.
        if constexpr (!MDP::is_model_eigen_v<M>)
            immediateRewards_ = computeImmediateRewards(pomdp);

        // Reset tolerance to set parameter;
        tolerance_ = initialTolerance_;

        // Helper methods
        BlindStrategies bs(infiniteHorizon, tolerance_);
        FastInformedBound fib(infiniteHorizon, tolerance_);
        PBVI pbvi(0, infiniteHorizon, tolerance_);

        // Here we use the BlindStrategies in order to obtain a very simple
        // initial lower bound.
        VList lbVList = std::get<1>(bs(pomdp, true));
        lbVList.erase(extractDominated(std::begin(lbVList), std::end(lbVList), unwrap), std::end(lbVList));

        auto lbBeliefs = std::vector<Belief>{initialBelief};

        // The same we do here with FIB for the input POMDP.
        MDP::QFunction ubQ = std::get<1>(fib(pomdp));
        AI_LOGGER(AI_SEVERITY_DEBUG, "Initial QFunction:\n" << ubQ);

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
        UpperBoundValueFunction ubV = {
            {initialBelief}, {fibQ.row(pomdp.getS()).maxCoeff()}
        };

        // We also store two numbers for the overall lowerBound/upperBound
        // differences. They are the values of the lowerBound and the
        // upperBound at the initial belief.
        double lb;
        findBestAtPoint(initialBelief, std::begin(lbVList), std::end(lbVList), &lb, unwrap);

        double ub = ubV.second[0];

        AI_LOGGER(AI_SEVERITY_INFO, "Initial bounds: " << lb << ", " << ub);

        while (true) {
            double threshold = std::pow(10, std::ceil(std::log10(std::max(std::fabs(ub), std::fabs(lb))))-precisionDigits_);
            auto var = ub - lb;

            if (checkEqualSmall(var, 0.0) || var < threshold)
                break;

            tolerance_ = threshold * (1.0 - pomdp.getDiscount()) / 2.0;
            // Now we find beliefs for both lower and upper bound where we
            // think we can improve. For the ub beliefs we also return their
            // values, since we need them to improve the ub.
            auto [newLbBeliefs, newUbBeliefs, newUbVals] = selectReachableBeliefs(pomdp, initialBelief, lbVList, lbBeliefs, ubQ, ubV);
            const auto newLbBeliefsSize = newLbBeliefs.size();
            const auto newUbBeliefsSize = newUbBeliefs.size();

            if (newLbBeliefsSize > 0) {
                AI_LOGGER(AI_SEVERITY_DEBUG, "LB: Adding " << newLbBeliefsSize << " new beliefs...");
                for (const auto & b : newLbBeliefs)
                    AI_LOGGER(AI_SEVERITY_DEBUG, "LB: - Belief: " << b.transpose());
                // If we found something interesting for the lower bound, we
                // add it to the beliefs we already had, and we rerun PBVI.
                lbBeliefs.insert(std::end(lbBeliefs), std::make_move_iterator(std::begin(newLbBeliefs)), std::make_move_iterator(std::end(newLbBeliefs)));

                {
                    // Then we remove all beliefs which don't actively support any
                    // alphaVectors.
                    auto sol = pbvi(pomdp, lbBeliefs, ValueFunction{std::move(lbVList)});

                    lbVList = std::move(std::get<1>(sol).back());

                    const auto rbegin = std::begin(lbVList);
                    const auto rend   = std::end  (lbVList);

                    lbBeliefs.erase(
                        extractBestUsefulPoints(
                            std::begin(lbBeliefs), std::end(lbBeliefs),
                            rbegin, rend, unwrap
                        ),
                        std::end(lbBeliefs)
                    );

                    // And we recompute the lower bound.
                    findBestAtPoint(initialBelief, rbegin, rend, &lb, unwrap);
                }
            }

            if (newUbBeliefsSize > 0) {
                // Here we do the same for the upper bound.
                const auto prevRows = pomdp.getS() + ubV.first.size();
                fibQ.conservativeResize(prevRows + newUbBeliefsSize, Eigen::NoChange);

                AI_LOGGER(AI_SEVERITY_DEBUG, "UB: Adding " << newUbBeliefsSize << " new beliefs...");
                for (size_t i = 0; i < newUbBeliefsSize; ++i)
                    AI_LOGGER(AI_SEVERITY_DEBUG, "UB: - Belief: " << newUbBeliefs[i].transpose() << " -- value: " << newUbVals[i]);

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
                fibQ = std::get<1>(fib(newPOMDP, newPOMDPSOSA, std::move(fibQ)));
                // We extract from the found upper bound the part for the
                // states of the input POMDP, and we copy them to our
                // upperBound alphavectors. We additionally update the values
                // for all ub beliefs.
                ubQ.noalias() = fibQ.topRows(pomdp.getS());
                for (size_t i = 0; i < ubV.second.size(); ++i)
                    ubV.second[i] = fibQ.row(pomdp.getS() + i).maxCoeff();

                // Finally, we remove some unused stuff, and we recompute the upperbound.
                cleanUp(ubQ, &ubV, &fibQ);

                ub = std::get<0>(LPInterpolation(initialBelief, ubQ, ubV));
            }

            // Update the difference between upper and lower bound so we can
            // return it/use it to stop the loop.
            auto oldVar = var;
            var = ub - lb;
            AI_LOGGER(AI_SEVERITY_INFO, "Updated bounds to " << lb << ", " << ub << " -- size LB: " << lbVList.size() << ", size UB " << ubV.first.size());

            // Stop if we didn't find anything new, or if we have converged the bounds.
            if (newLbBeliefsSize + newUbBeliefsSize == 0 || std::fabs(var - oldVar) < tolerance_ * 5)
                break;
        }
        return std::make_tuple(lb, ub, lbVList, ubQ);
    }

    template <typename M, typename>
    std::tuple<GapMin::IntermediatePOMDP, SparseMatrix4D> GapMin::makeNewPomdp(const M& model, const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV) {
        size_t S = model.getS() + ubV.first.size();

        // First we build the new reward function. For normal states, this is
        // the same as the old one. For all additional states (beliefs), we
        // simply take their expected reward with respect to the original
        // reward function.
        Matrix2D R(S, model.getA());
        const auto & ir = [&]{
            if constexpr (MDP::is_model_eigen_v<M>) return model.getRewardFunction();
            else return immediateRewards_;
        }();

        R.topRows(model.getS()) = ir;
        for (size_t b = 0; b < ubV.first.size(); ++b)
            R.row(model.getS()+b) = ubV.first[b].transpose() * ir;

        // Now we create the SOSA matrix for this new POMDP. For each pair of
        // action/observation, and for each belief we have (thus state), we
        // compute the probability of going to any other belief.
        //
        // This is done through the UB function, although I must admit I don't
        // fully understand the math behind of why it works.
        Belief helper(model.getS()), corner(model.getS());
        corner.setZero();

        SparseMatrix4D sosa( boost::extents[model.getA()][model.getO()] );
        const auto updateMatrix = [&](SparseMatrix2D & m, const Belief & b, size_t a, size_t o, size_t index) {
            updateBeliefUnnormalized(model, b, a, o, &helper);
            auto sum = helper.sum();
            if (checkDifferentSmall(sum, 0.0)) {
                // Note that we do not normalize helper since we'd also have to
                // multiply `dist` by the same probability. Instead we don't
                // normalize, and we don't multiply, so we save some work.
                Vector dist = std::get<1>(LPInterpolation(helper, ubQ, ubV));
                for (size_t i = 0; i < S; ++i)
                    if (checkDifferentSmall(dist[i], 0.0))
                        m.insert(index, i) = dist[i];
            }
        };

        for (size_t a = 0; a < model.getA(); ++a) {
            for (size_t o = 0; o < model.getO(); ++o) {
                SparseMatrix2D m(S, S);
                for (size_t s = 0; s < model.getS(); ++s) {
                    corner[s] = 1.0;
                    updateMatrix(m, corner, a, o, s);
                    corner[s] = 0.0;
                }

                for (size_t b = 0; b < ubV.first.size(); ++b)
                    updateMatrix(m, ubV.first[b], a, o, model.getS() + b);

                // After updating all rows of the matrix, we put it inside the
                // SOSA matrix.
                sosa[a][o] = std::move(m);
                sosa[a][o].makeCompressed();
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

    template <typename M, typename>
    std::tuple<std::vector<Belief>, std::vector<Belief>, std::vector<double>> GapMin::selectReachableBeliefs(
            const M & pomdp, const Belief & initialBelief, const VList & lbVList,
            const std::vector<Belief> & lbBeliefs, const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
        )
    {
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
        {
            double currentLowerBound;
            const auto rbegin = std::begin(lbVList);
            const auto rend   = std::end  (lbVList);
            findBestAtPoint(initialBelief, rbegin, rend, &currentLowerBound, unwrap);
            const double currentUpperBound = std::get<0>(LPInterpolation(initialBelief, ubQ, ubV));
            queue.emplace(QueueElement(initialBelief, 0.0, 1.0, currentLowerBound, currentUpperBound, 1, {}));
        }

        while (!queue.empty() && newBeliefs < maxNewBeliefs) {
            const auto [belief, gap, beliefProbability, currentLowerBound, currentUpperBound, depth, path] = queue.top();
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
            const auto & ir = [&]{
                if constexpr (MDP::is_model_eigen_v<M>) return pomdp.getRewardFunction();
                else return immediateRewards_;
            }();
            const auto [ubAction, ubActionValue] = bestPromisingAction(pomdp, ir, belief, ubQ, ubV);
            const auto [lbAction, lbActionValue] = bestConservativeAction(pomdp, ir, belief, lbVList);

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
                if (std::any_of(std::begin(newUbBeliefs), std::end(newUbBeliefs), check))
                    return false;
                if (std::any_of(std::begin(ubV.first), std::end(ubV.first), check))
                    return false;
                return true;
            };

            if (validForUb(belief) && ubActionValue < currentUpperBound - tolerance_) {
                newUbBeliefs.push_back(belief);
                newUbValues.push_back(ubActionValue);

                // Find all beliefs that brought us here we didn't already have.
                // Again, we don't consider corners.
                for (const auto & p : path) {
                    if (validForUb(p)) {
                        newUbBeliefs.push_back(p);
                        newUbValues.push_back(std::get<0>(LPInterpolation(p, ubQ, ubV)));
                    }
                }
                // Note we only count a single belief even if we added more via
                // the path (as per original code).
                ++newBeliefs;
            }

            /***********************
             **     LOWER GAP     **
             ***********************/

            // For the lower gap we don't care about corners (as per original
            // code). We still check on the lower bound lists though.
            const auto validForLb = [&newLbBeliefs, &lbBeliefs](const Belief & b) {
                const auto check = [&b](const Belief & bb){ return checkEqualProbability(b, bb); };
                if (std::any_of(std::begin(newLbBeliefs), std::end(newLbBeliefs), check))
                    return false;
                if (std::any_of(std::begin(lbBeliefs), std::end(lbBeliefs), check))
                    return false;
                return true;
            };

            if (validForLb(belief) && lbActionValue > currentLowerBound + tolerance_) {
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
                Belief nextBelief = updateBeliefPartialUnnormalized(pomdp, intermediateBelief, ubAction, o);

                const auto nextBeliefProbability = nextBelief.sum();
                if (checkEqualSmall(nextBeliefProbability, 0.0)) continue;
                nextBelief /= nextBeliefProbability;

                const auto check = [&nextBelief](const Belief & bb){ return checkEqualProbability(nextBelief, bb); };
                if (std::any_of(std::begin(visitedBeliefs), std::end(visitedBeliefs), check)) continue;

                const double ubValue = std::get<0>(LPInterpolation(nextBelief, ubQ, ubV));
                double lbValue;
                findBestAtPoint(nextBelief, std::begin(lbVList), std::end(lbVList), &lbValue, unwrap);

                if ((ubValue - lbValue) * std::pow(pomdp.getDiscount(), depth) > tolerance_ * 20) {
                    const auto nextBeliefOverallProbability = nextBeliefProbability * beliefProbability * pomdp.getDiscount();
                    const auto nextBeliefGap = nextBeliefOverallProbability * (ubValue - lbValue);

                    const auto qcheck = [&nextBelief](const QueueElement & qe){ return checkEqualProbability(nextBelief, std::get<0>(qe)); };
                    const auto it = std::find_if(std::begin(queue), std::end(queue), qcheck);
                    if (it == std::end(queue)) {
                        queue.emplace(
                                std::move(nextBelief),
                                nextBeliefGap,
                                nextBeliefOverallProbability,
                                lbValue,
                                ubValue,
                                depth+1,
                                newPath
                        );
                    } else {
                        auto handle = QueueType::s_handle_from_iterator(it);
                        std::get<1>(*handle) += nextBeliefGap;
                        std::get<2>(*handle) += nextBeliefOverallProbability;
                        std::get<5>(*handle) = std::min(std::get<5>(*handle), depth+1);
                        queue.increase(handle);
                    }
                }
            }
        }
        return std::make_tuple(std::move(newLbBeliefs), std::move(newUbBeliefs), std::move(newUbValues));
    }
}

#endif
