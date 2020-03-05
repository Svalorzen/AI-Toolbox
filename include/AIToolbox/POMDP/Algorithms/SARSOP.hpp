#ifndef AI_TOOLBOX_POMDP_SARSOP_HEADER_FILE
#define AI_TOOLBOX_POMDP_SARSOP_HEADER_FILE

#include <AIToolbox/Impl/Logging.hpp>

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/TypeTraits.hpp>

#include <AIToolbox/POMDP/Algorithms/BlindStrategies.hpp>
#include <AIToolbox/POMDP/Algorithms/FastInformedBound.hpp>

namespace AIToolbox::POMDP {
    /**
     * @brief This class implements the SARSOP algorithm.
     *
     * This algorithm works by computing lower and upper bounds on what is
     * believed to be the optimal policy.
     *
     * SARSOP tries to keep computational costs in check by only computing
     * alphavectors and upper bounds by exploring future action/observation
     * pairs which are believed to fall in the path of the optimal policy.
     *
     * Since at the start the optimal policy is not known, SARSOP employs a
     * series of heuristics to ensure that the paths it explores are indeed
     * correct. At the same time, it also aggressively prunes the found
     * alphavectors and beliefs to keep further exploration cheap.
     *
     * The result should be lower/upper bounds that are reasonably close to
     * optimal as long as one remains in the part of the belief space reachable
     * via the optimal policy. Once a non-optimal action is taken, the bounds
     * are likely to be loose.
     */
    class SARSOP {
        public:
            /**
             * @brief Basic constructor.
             *
             * @param tolerance The tolerance to reach when solving a POMDP.
             * @param delta The initial delta to use for pruning.
             */
            SARSOP(double tolerance, double delta = 0.1);

            /**
             * @brief This function sets the tolerance to reach when solving a POMDP.
             *
             * @param tolerance The new tolerance.
             */
            void setTolerance(double tolerance);

            /**
             * @brief This function returns the currently set tolerance to reach when solving a POMDP.
             *
             * @return The current tolerance.
             */
            double getTolerance() const;

            void setDelta(double delta);
            double getDelta() const;

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
            /**
             * @brief
             */
            struct TreeNode {
                Belief belief;

                // Number of non-suboptimal branches that reach this belief.
                unsigned count;

                // Bounds info
                double UB, LB;
                size_t actionUb;
                // Per action info (per row: immediate reward, UB, suboptimal)
                // Only initialized during expand
                Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign> actionData;

                // Per action-observation info
                // Only initialized during expand
                struct Children {
                    size_t id;
                    double observationProbability;
                };
                boost::multi_array<Children, 2> children;
            };

            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void samplePoints(
                const M & pomdp,
                const VList & lbV,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
            );

            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void expandLeaf(
                size_t id, const M & model,
                const VList & lbV,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
            );

            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void updateNode(
                TreeNode & node, const M & model,
                const VList & lbV,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV,
                bool expand
            );

            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void backupNode(
                size_t id, const M & model,
                VList & lbV,
                MDP::QFunction & ubQ, UpperBoundValueFunction & ubV
            );

            double predictValue(size_t id, const TreeNode & node);
            void deltaPrune(VList & lbV);
            void deltaUpdate(const VList & lbV);
            void updateSubOptimalPaths(TreeNode & root);
            void treePrune(TreeNode & root);
            void treeRevive(TreeNode & root);

            class LBPredictor {
                public:
                    LBPredictor(size_t entropyBins, size_t UBBins, const MDP::QFunction & ubQ);

                    //       bin avg, bin err
                    std::pair<double, double> predict(size_t id, const TreeNode & node);

                private:
                    struct Bin {
                        double avg;
                        double error;
                        unsigned count;
                    };

                    const Bin & update(size_t id, const TreeNode & node);

                    const MDP::QFunction & ubQ_;
                    size_t entropyBins_, UBBins_;
                    double entropyStep_, UBMin_, UBStep_;

                    //                 id             initialized, ei,     ubi,    lb,    err
                    std::unordered_map<size_t, std::tuple<bool,    size_t, size_t, double, double>> nodes_;
                    boost::multi_array<Bin, 2> bins_;
            };

            double tolerance_, initialDelta_;

            // Data reset at each main call
            double delta_;
            Matrix2D immediateRewards_;
            std::vector<TreeNode> treeStorage_;
            // We use this to check whether we have already encountered a
            // Belief or not. Note that this is very sensitive to floating
            // point errors, so it's probably not the best way to go about it
            // (maybe an std::map using lexicographical order might be better).
            // At the same time, SARSOP original code converted Beliefs to
            // strings and applied md5 hashing to them, so it probably can't be
            // worse than that either.
            std::unordered_map<Belief, size_t, boost::hash<Belief>> beliefToNode_;
            std::vector<LBPredictor> predictors_;

            // Storage to avoid reallocations
            std::vector<size_t> sampledNodes_;
            std::vector<char> backuppedActions_;
            Belief intermediateBeliefTmp_, nextBeliefTmp_;
    };

    template <typename M, typename>
    std::tuple<double, double, VList, MDP::QFunction> SARSOP::operator()(const M & pomdp, const Belief & initialBelief) {
        constexpr unsigned infiniteHorizon = 1000000;

        // ##############################
        // ### Resetting general data ###
        // ##############################

        // Reset delta to the initial parameter set.
        delta_ = initialDelta_;

        // Cache immediate rewards if we can't read the reward function directly.
        if constexpr (!MDP::is_model_eigen_v<M>)
            immediateRewards_ = computeImmediateRewards(pomdp);
        const auto & ir = [&]{
            if constexpr (MDP::is_model_eigen_v<M>) return pomdp.getRewardFunction();
            else return immediateRewards_;
        }();

        // First allocation for root node & children
        treeStorage_.clear();
        treeStorage_.reserve(pomdp.getA() * pomdp.getO() + 1);

        beliefToNode_.clear();

        // Bins initialization. Note that the "multiple bin predictors"
        // mechanism has been taken from the original author's code, as the
        // paper itself does not mention it. Just modify the constants below if
        // you want the bins to behave differently.
        predictors_.clear();

        // ########################################
        // ### Pre-allocating temporary storage ###
        // ########################################

        backuppedActions_.resize(pomdp.getA());
        intermediateBeliefTmp_.resize(pomdp.getS());
        nextBeliefTmp_.resize(pomdp.getS());

        // ################################
        // ### Computing initial bounds ###
        // ################################

        // Helper methods to compute initial LB/UB. Since SARSOP is not really
        // the best method, it's unlikely that we will ask very stringent
        // tolerances (unless we want to wait a while). However, since these
        // two methods are pretty fast, there's not harm in asking them a
        // tighter tolerance if they can reach it.. if they can't they'll
        // simply stop on their own.
        BlindStrategies bs(infiniteHorizon, std::min(0.00001, tolerance_));
        FastInformedBound fib(infiniteHorizon, std::min(0.00001, tolerance_));

        // Here we use the BlindStrategies in order to obtain a very simple
        // initial lower bound.
        VList lbVList = std::get<1>(bs(pomdp, true));
        lbVList.erase(extractDominated(std::begin(lbVList), std::end(lbVList), unwrap), std::end(lbVList));

        // ### Delta Pruning Setup ###
        //
        // In order to efficiently and aggressively prune alphavectors, SARSOP
        // prunes one when it is dominated across the whole belief space we
        // have explored, i.e. all the beliefs in treeStorage_.
        //
        // However, checking every vector every time against all beliefs would
        // be a bit expensive. So what we do (as in the original code), is
        // instead we associate each alphavector with a set of witness points.
        // If it is dominated over that, it's dead. This idea comes from the
        // HVSI paper.
        //
        // Additionally, we keep all vectors which are the max at any given belief.
        //
        // Since we need to keep track of all these points per each
        // alphavector, we store them (temporarily) in the observations field
        // of each VEntry. This is because we are not going to use that field
        // for anything else here, and so might as well re-use it.
        //
        // In particular, each vector will be in this form:
        //
        // [ number_of_max_points + 1, max_point_id, ..., witness_point_id, ... ]
        //
        // The first element is simply to keep track of the "end" range of the
        // max_point_ids, so we can remember which id means what. The id ranges
        // are kept independently sorted, and refer to the id of the respective
        // TreeNode in treeStorage_.
        //
        // Additionally, each id is actually the id plus |S|. This is because
        // we also want to store the corners, so what we do is that ids in [0,
        // S) mean corners, while ids higher refer to the (id-S) element in
        // treeStorage_. You can see all this in action in the deltaPrune
        // function.
        //
        // In any case, here we have to setup the vectors for the corners and
        // the initial belief, so they are ready to go.

        // First element means that the end of the range of the max-ids is 1,
        // i.e. we have no maxes (nor witnesses atm).
        for (auto & ve : lbVList)
            ve.observations.push_back(1);

        // For each corner, find the best alphavector for it, and add the
        // corner in its max list.
        for (size_t s = 0; s < pomdp.getS(); ++s) {
            auto it = findBestAtSimplexCorner(s, std::begin(lbVList), std::end(lbVList), nullptr, unwrap);
            // Mark that we are adding a max to the list
            ++it->observations[0];
            // Add the corner to the list (we can do push_back since we are
            // sure we have no witness points yet, so we don't mix them).
            it->observations.push_back(s);
        }
        // Finally, find the max for initialBelief and assign that.
        auto it = findBestAtPoint(initialBelief, std::begin(lbVList), std::end(lbVList), nullptr, unwrap);
        ++it->observations[0];
        // id of initialBelief is 0, since not a corner => 0 + S = S
        it->observations.push_back(pomdp.getS());

        // The same we do here with FIB for the input POMDP.
        MDP::QFunction ubQ = std::get<1>(fib(pomdp));
        AI_LOGGER(AI_SEVERITY_DEBUG, "Initial QFunction:\n" << ubQ);

        // While we store the lower bound as alphaVectors, the upper bound is
        // composed by both alphaVectors (albeit only S of them - out of the
        // FastInformedBound), and a series of belief-value pairs, which we'll
        // use with the later-constructed new POMDP in order to improve our
        // bounds.
        UpperBoundValueFunction ubV = {
            {initialBelief}, {(initialBelief.transpose() * ubQ).maxCoeff()}
        };

        // ###########################
        // ### Setup UB predictors ###
        // ###########################

        // This we use to estimate the UB buckets for each belief.
        const auto initialUbQ = ubQ;

        constexpr unsigned numBins = 2;
        constexpr unsigned entropyBins = 5;
        constexpr unsigned ubBins = 5;
        constexpr unsigned binScaling = 2;

        for (unsigned i = 0; i < numBins; ++i) {
            const unsigned scaling = std::pow(binScaling, i);
            predictors_.emplace_back(entropyBins * scaling, ubBins * scaling, initialUbQ);
        }

        // #######################
        // ### Setup tree root ###
        // #######################

        treeStorage_.emplace_back();

        // Note that we can't make a reference alias to the root since
        // treeStorage_ is going to reallocate multiple times during solving.
        treeStorage_[0].belief = initialBelief;
        treeStorage_[0].count = 1;
        updateNode(treeStorage_[0], pomdp, lbVList, ubQ, ubV, false);

        AI_LOGGER(AI_SEVERITY_INFO, "Initial bounds: " << treeStorage_[0].LB << ", " << treeStorage_[0].UB);
        AI_LOGGER(AI_SEVERITY_INFO, "Root UBs: " << treeStorage_[0].actionData.row(1));

        // ##################
        // ### Begin work ###
        // ##################

        while (true) {
            AI_LOGGER(AI_SEVERITY_INFO, "Sampling...");
            // Deep sample a branch of the action/observation trees. The
            // sampled nodes (except the last one where we stop) are added to
            // sampledNodes_.
            samplePoints(pomdp, lbVList, ubQ, ubV);
            AI_LOGGER(AI_SEVERITY_INFO, "Sampled " << sampledNodes_.size() << " beliefs.");

            // If we have no nodes it means we stopped at the root, so we have
            // already shrinked the gap enough; we are done.
            if (sampledNodes_.size() == 0)
                break;

            AI_LOGGER(AI_SEVERITY_INFO, "Backing up...");
            // Backup the nodes we sampled, from (node-before) leaf to root.
            // This updates the lower and upper bounds by adding
            // alphavectors/points to them.
            for (auto rIt = std::rbegin(sampledNodes_); rIt != std::rend(sampledNodes_); ++rIt)
                backupNode(*rIt, pomdp, lbVList, ubQ, ubV);

            // # Lower Bound Pruning #

            AI_LOGGER(AI_SEVERITY_INFO, "Delta pruning... (" << lbVList.size() << " alphavectors)");
            // We aggressively prune the lbVList based on the beliefs we have
            // explored. This prunes both using direct dominance as well as
            // delta dominance, i.e. vectors count as dominated if they are
            // dominated within a given neighborhood of all their witness
            // beliefs.
            deltaPrune(lbVList);
            AI_LOGGER(AI_SEVERITY_INFO, "Pruned (now " << lbVList.size() << " alphavectors)");

            // # Upper Bound Pruning #

            AI_LOGGER(AI_SEVERITY_INFO, "UB pruning...");
            // Prune unused beliefs that do not contribute to the upper bound.
            // This means that their value is *higher* than what we can
            // approximate using the other beliefs.
            size_t i = ubV.first.size();
            do {
                --i;

                // We swap the current belief to check at the end, and we
                // temporarily remove it so we can test the interpolation
                // without it.
                std::swap(ubV.first[i], ubV.first.back());
                std::swap(ubV.second[i], ubV.second.back());

                auto belief = std::move(ubV.first.back());
                auto value = ubV.second.back();

                ubV.first.pop_back();
                ubV.second.pop_back();

                // If its original value is lower than the interpolation, we
                // still need it to improve our upper bound.
                if (value < std::get<0>(sawtoothInterpolation(belief, ubQ, ubV))) {
                    // Thus, we put it back inside.
                    ubV.first.emplace_back(std::move(belief));
                    ubV.second.emplace_back(value);
                }
            } while (i != 0 && ubV.first.size() > 1);

            if (treeStorage_[0].UB - treeStorage_[0].LB <= tolerance_)
                break;
        }

        // Remove witness data from lbVList since we don't need to pass it
        // outside.
        for (auto & ventry : lbVList)
            ventry.observations.clear();

        return std::make_tuple(treeStorage_[0].LB, treeStorage_[0].UB, lbVList, ubQ);
    }

    template <typename M, typename>
    void SARSOP::samplePoints(const M & pomdp, const VList & lbVList, const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV) {
        sampledNodes_.clear();
        // Always begin sampling from the root. We are going to go down a path
        // until we hit our stopping conditions. If we end up outside the tree,
        // we are going to add the new nodes to it as we go along.
        size_t currentNodeId = 0;
        const double rootGap = treeStorage_[0].UB - treeStorage_[0].LB;

        int depth = 0;
        double L = treeStorage_[0].LB;
        double U = treeStorage_[0].UB;

        while (true) {
            // Compute target gap for this depth.
            const double targetGap = rootGap * std::pow(pomdp.getDiscount(), -depth);

            {
                // Here we check whether we should stop. Note that the
                // reference to node is intentionally kept scoped, as we may
                // need to expand this node later, and doing so will invalidate
                // its address.
                const TreeNode & node = treeStorage_[currentNodeId];

                const double finalExcess = node.UB - node.LB - 0.5 * targetGap;
                if (finalExcess <= 0.0)
                    break;

                // Stopping condition; we stop sampling if either our approximation
                // falls below the lower bound, or if our upper bound is too low.
                const auto vHat = predictValue(currentNodeId, node);
                if (vHat <= L && node.UB <= std::max(U, node.LB + targetGap))
                    break;
            }

            // We are indeed going down this node, so we add it to the nodes
            // sampled.
            AI_LOGGER(AI_SEVERITY_DEBUG, "Accepted node " << currentNodeId << " for sampling.");
            sampledNodes_.push_back(currentNodeId);

            AI_LOGGER(AI_SEVERITY_DEBUG, "Node is leaf, expanding...");
            // Precompute this node's children if it was a leaf.
            if (treeStorage_[currentNodeId].children.size() == 0)
                expandLeaf(currentNodeId, pomdp, lbVList, ubQ, ubV);

            // Now we can take a reference as we won't need to allocate again.
            const TreeNode & node = treeStorage_[currentNodeId];
            AI_LOGGER(AI_SEVERITY_DEBUG, "Sampling belief " << node.belief.transpose() << " with id " << currentNodeId);

            // Otherwise we keep sampling.
            const auto L1 = std::max(L, node.LB);
            const auto U1 = std::max(U, node.LB + targetGap);

            AI_LOGGER(AI_SEVERITY_DEBUG, "a1, o1...");
            // FIXME: possible do randomization
            const auto a1 = node.actionUb;
            // FIXME: possible do randomization
            size_t o1 = 0;
            {
                const double nextDepthGap = targetGap / pomdp.getDiscount();
                double maxVal = std::numeric_limits<double>::lowest();
                for (size_t o = 0; o < pomdp.getO(); ++o) {
                    if (node.children[a1][o].observationProbability == 0.0) continue;

                    const auto & childNode = treeStorage_[node.children[a1][o].id];
                    const auto val = (childNode.UB - childNode.LB - nextDepthGap) * node.children[a1][o].observationProbability;
                    if (val > maxVal) {
                        maxVal = val;
                        o1 = o;
                    }
                }
            }

            AI_LOGGER(AI_SEVERITY_DEBUG, "Lnorm, Unorm...");
            double Lnorm = 0.0, Unorm = 0.0;
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                if (o == o1) continue;

                const auto & childNode = treeStorage_[node.children[a1][o].id];

                Lnorm += childNode.LB * node.children[a1][o].observationProbability;
                Unorm += childNode.UB * node.children[a1][o].observationProbability;
            }

            AI_LOGGER(AI_SEVERITY_DEBUG, "Lt, Ut...");
            // Lt, Ut
            L = ((L1 - node.actionData(0, a1)) / pomdp.getDiscount() - Lnorm) / node.children[a1][o1].observationProbability;
            U = ((U1 - node.actionData(0, a1)) / pomdp.getDiscount() - Unorm) / node.children[a1][o1].observationProbability;

            // Set the new node to go down to.
            currentNodeId = node.children[a1][o1].id;

            ++depth;
        }
    }

    template <typename M, typename>
    void SARSOP::expandLeaf(
            const size_t id, const M & pomdp,
            const VList & lbVList,
            const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
        )
    {
        TreeNode * nodep = &treeStorage_[id];

        assert(node->children.size() == 0);
        // This assert is to say that we shouldn't really be going down a
        // provenly suboptimal path, so this should not really happen.  If it
        // happens, it might be something is broken or I misunderstood
        // something.
        assert(node->count > 0);

        AI_LOGGER(AI_SEVERITY_DEBUG, "Updating node...");
        // Allocate precompute bound values for future backups
        updateNode(*nodep, pomdp, lbVList, ubQ, ubV, true);

        AI_LOGGER(AI_SEVERITY_DEBUG, "Allocating children...");
        // Allocate children memory
        nodep->children.resize(boost::extents[pomdp.getA()][pomdp.getO()]);

        AI_LOGGER(AI_SEVERITY_DEBUG, "Starting loop..");
        for (size_t a = 0; a < pomdp.getA(); ++a) {
            updateBeliefPartial(pomdp, nodep->belief, a, &intermediateBeliefTmp_);

            for (size_t o = 0; o < pomdp.getO(); ++o) {
                AI_LOGGER(AI_SEVERITY_DEBUG, "a = " << a << "; o = " << o);
                auto & child = nodep->children[a][o];

                updateBeliefPartialUnnormalized(pomdp, intermediateBeliefTmp_, a, o, &nextBeliefTmp_);

                const auto prob = nextBeliefTmp_.sum();

                if (checkEqualSmall(prob, 0.0)) {
                    // observationProbability for this child is 0.0 by default,
                    // we'll use that for future checks.
                    continue;
                }
                nextBeliefTmp_ /= prob;

                child.observationProbability = prob;

                AI_LOGGER(AI_SEVERITY_DEBUG, "Looking for it: " << nextBeliefTmp_.transpose());

                const auto it = beliefToNode_.find(nextBeliefTmp_);
                if (it != beliefToNode_.end()) {
                    AI_LOGGER(AI_SEVERITY_DEBUG, "Found, setting id and continuing...");
                    // If the node already existed, we simply point to it, and
                    // increase its reference count.
                    child.id = it->second;
                    if (++treeStorage_[child.id].count == 1) {
                        AI_LOGGER(AI_SEVERITY_DEBUG, "Reviving node " << child.id << "...");
                        // If it's count was 0 before, then it represented a
                        // previously pruned branch. Since it's now back in the
                        // tree, we need to "revive" all its children warning
                        // them that a new path to them is open.
                        //
                        // Note that this does not bring "directly" alive any
                        // alphavectors associated with those beliefs (as
                        // alphavectors of dead branches are pruned away), but
                        // we'll have to wait until direct exploration makes us
                        // do backup of those beliefs again.
                        treeRevive(treeStorage_[child.id]);
                    }
                    continue;
                }

                // Finish storing info about child as its reference is about to
                // go stale.
                AI_LOGGER(AI_SEVERITY_DEBUG, "Not found, Setting new id...");
                child.id = treeStorage_.size();
                AI_LOGGER(AI_SEVERITY_DEBUG, "ID = " << child.id);
                beliefToNode_[nextBeliefTmp_] = child.id;

                AI_LOGGER(AI_SEVERITY_DEBUG, "Emplacing in storage... " << treeStorage_.size());
                // Adding a node to treeStorage_ invalidates every single
                // reference we are holding to anything in it, since it may
                // reallocate. Keep it in mind.
                treeStorage_.emplace_back();
                // Re-assign to nodep to get the possibly new pointer.
                nodep = &treeStorage_[id];

                AI_LOGGER(AI_SEVERITY_DEBUG, "Setting node properties...");
                auto & childNode = treeStorage_.back();

                childNode.belief = nextBeliefTmp_;
                childNode.count = 1;
                // Compute UB and LB for this child
                AI_LOGGER(AI_SEVERITY_DEBUG, "Updating new leaf...");
                updateNode(childNode, pomdp, lbVList, ubQ, ubV, false);
            }
        }
    }

    template <typename M, typename>
    void SARSOP::updateNode(
            TreeNode & node, const M & pomdp,
            const VList & lbVList,
            const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV,
            bool expand
        )
    {
        const auto & ir = [&]{
            if constexpr (MDP::is_model_eigen_v<M>) return pomdp.getRewardFunction();
            else return immediateRewards_;
        }();
        // We update the UB using the sawtooth approximation since it's work we
        // have to do whether we are expanding a node or updating a leaf.
        Vector ubs; // Here we store per-action upper-bounds in case we need them.
        const auto ub = bestPromisingAction<false>(pomdp, ir, node.belief, ubQ, ubV, &ubs);
        node.UB = std::get<1>(ub);
        node.actionUb = std::get<0>(ub);

        if (expand) {
            // If we are expanding the node, we are only really interested in the
            // actionData, as it contains pre-computed data which allows us to
            // possibly skip some work when doing upper-bound backups.
            node.actionData.resize(Eigen::NoChange, pomdp.getA());
            node.actionData.row(0) = node.belief.transpose() * ir;
            node.actionData.row(1) = ubs;
            node.actionData.row(2).fill(0);
        } else {
            // Otherwise, we are just computing the upper and lower bounds of a
            // leaf node. The UB we already did, so here we do the LB.
            const auto lb = bestConservativeAction(pomdp, ir, node.belief, lbVList);
            node.LB = std::get<1>(lb);
        }
    }

    template <typename M, typename>
    void SARSOP::backupNode(size_t id, const M & pomdp, VList & lbVList, MDP::QFunction & ubQ, UpperBoundValueFunction & ubV) {
        const auto & ir = [&]{
            if constexpr (MDP::is_model_eigen_v<M>) return pomdp.getRewardFunction();
            else return immediateRewards_;
        }();

        TreeNode & node = treeStorage_[id];
        {
            // Update lower bound and extract a new alphavector.
            Vector alpha;
            const auto result = bestConservativeAction(pomdp, ir, node.belief, lbVList, &alpha);
            node.LB = std::get<1>(result);
            // Add new alphavector with its witness point inserted
            lbVList.emplace_back(std::move(alpha), std::get<0>(result), VObs{1, id + pomdp.getS()});
        }

        // For the upper bound we use the precomputed values to try to skip
        // some work. Since updating a upper-bound can only lower it, we update
        // only the highest value. If then it's still the highest, we are done.
        // Otherwise, we select the new highest and continue, until we end up
        // with a new max.
        std::fill(std::begin(backuppedActions_), std::end(backuppedActions_), false);
        auto maxAction = node.actionUb;

        while (!backuppedActions_[maxAction]) {
            double sum = 0.0;
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                const double obsP = node.children[maxAction][o].observationProbability;

                if (obsP == 0.0) continue;

                const auto & childNode = treeStorage_[node.children[maxAction][o].id];

                sum += obsP * std::get<0>(sawtoothInterpolation(childNode.belief, ubQ, ubV));
            }
            sum = node.actionData(0, maxAction) + pomdp.getDiscount() * sum;

            node.actionData(1, maxAction) = sum;
            backuppedActions_[maxAction] = true;

            node.UB = node.actionData.row(1).maxCoeff(&maxAction);
        }
        node.actionUb = maxAction;

        // Finally, we can add update this belief's value in the upper bound.
        // If it's a corner point, we modify ubQ directly; otherwise we just
        // add it to ubV.
        for (size_t s = 0; s < pomdp.getS(); ++s) {
            if (checkEqualSmall(node.belief[s], 1.0)) {
                ubQ(s, maxAction) = node.UB;
                return;
            }
        }
        ubV.first.push_back(node.belief);
        ubV.second.push_back(node.UB);
    }
}

#endif
