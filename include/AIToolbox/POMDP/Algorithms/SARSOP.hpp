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

            /**
             * @brief This function sets the delta for pruning to use at the start of a solving process.
             *
             * Note that during the solving process the delta is modified
             * dynamically based on heuristics.
             *
             * \sa deltaPrune()
             *
             * @param delta The new delta to use.
             */
            void setDelta(double delta);

            /**
             * @brief This function returns the delta for pruning to use at the start of a solving process.
             *
             * \sa deltaPrune()
             *
             * @return The currently set delta.
             */
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
             * @brief This struct represents a node in our Belief graph.
             *
             * From our initial Belief, given actions and observations, we are
             * going to end up in other Beliefs. This expands ideally into a
             * tree, but since we may visit certain Beliefs more than once,
             * it's actually a graph (sorry for the name).
             *
             * This struct contains the data we need for every Belief we
             * encounter: what the Belief is, whether it's suboptimal, it's
             * upper and lower bounds, and to what Beliefs we end up to with
             * certain action/observation pairs.
             *
             * This data is kept here to avoid having to recompute it all the
             * time.
             */
            struct TreeNode {
                Belief belief;

                // Number of non-suboptimal branches that reach this Belief.
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

            /**
             * @brief This function expands the Belief tree and finds nodes which should be backed up.
             *
             * This function selects the branches in the Belief tree which are
             * most likely to reduce the bound gap at the root, and explores
             * them. It then uses heuristics in order to decide when to stop.
             *
             * If needed, it expands new nodes and adds them to the internal
             * tree. All nodes sampled (until the very last leaf where we have
             * stopped) are added to the sampledNodes_ internal variable.
             *
             * @param pomdp The POMDP to solve.
             * @param lbV The current lower bound.
             * @param ubQ The QFunction containing the upper bound.
             * @param ubV The belief-value pairs for the upper bound.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void samplePoints(
                const M & pomdp,
                const VList & lbV,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
            );

            /**
             * @brief This function precomputes values and children for a given leaf.
             *
             * As we descend the tree, we need to explore new nodes. Once we
             * find that we need to descend into a leaf, we expand and add its
             * children to the tree.
             *
             * In addition, we precompute the actionData variable of the nodes,
             * as it will be useful during backup.
             *
             * @param id The id of the leaf to expand.
             * @param model The POMDP to solve.
             * @param lbV The current lower bound.
             * @param ubQ The QFunction containing the upper bound.
             * @param ubV The belief-value pairs for the upper bound.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void expandLeaf(
                size_t id, const M & model,
                const VList & lbV,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV
            );

            /**
             * @brief This function computes the bounds for a node.
             *
             * This function computes the bounds for a node, without modifying
             * the lower and upper bounds.
             *
             * In particular, updating the upper bound with this function is
             * more expensive than during backup, as during backup we do a
             * slight optimization to only compute it for the best action.
             *
             * This function has an additional "expand" parameter, to use when
             * we want to update a node that is being expanded. If that's the
             * case, we only update the lower bound, but we also initialize the
             * actionData matrix, which will be used during backup.
             *
             * If a node is being expanded we do not update the lower bound, as
             * we are trying to do the minimum work required.
             *
             * @param node The node to update.
             * @param model The POMDP to solve.
             * @param lbV The current lower bound.
             * @param ubQ The QFunction containing the upper bound.
             * @param ubV The belief-value pairs for the upper bound.
             * @param expand Whether we are expanding this node or not.
             */
            template <typename M, typename = std::enable_if_t<is_model_v<M>>>
            void updateNode(
                TreeNode & node, const M & model,
                const VList & lbV,
                const MDP::QFunction & ubQ, const UpperBoundValueFunction & ubV,
                bool expand
            );

            /**
             * @brief This function performs a backup on the specified node of the tree.
             *
             * Note that only expanded (i.e. non-leaf) nodes can be backed up.
             *
             * The node will get it's lower and upper bound updated, and from
             * them we will add a new alphavector to the lower bound, and a new
             * belief-point pair to the upper bound (possibly updating ubQ in
             * case the node is a corner of the simplex).
             *
             * @param id The id of a non-leaf node.
             * @param model The POMDP to solve.
             * @param lbV The current lower bound.
             * @param ubQ The QFunction containing the upper bound.
             * @param ubV The belief-value pairs for the upper bound.
             */
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

            /**
             * @brief This class predicts the value of a TreeNode based on entropy and upper bound.
             */
            class LBPredictor {
                public:
                    /**
                     * @brief Basic constructor.
                     *
                     * @param entropyBins The number of bins to use for entropy.
                     * @param UBBins The number of bins to use for the upper bound.
                     * @param ubQ A reference to the initial upper bound surface (the one obtained through FastInformedBound).
                     */
                    LBPredictor(size_t entropyBins, size_t UBBins, const MDP::QFunction & ubQ);

                    /**
                     * @brief This function predicts the value of the input node.
                     *
                     * If the node is new, we automatically initialize it and
                     * add it to its correct bucket.
                     *
                     * We average the lower bound values of the nodes in the
                     * same bucket, and that's our predicted value.
                     *
                     * An exception is made for nodes which are alone in their
                     * bucket; in that case their upper bound is returned (with
                     * error 0).
                     *
                     * @param id The unique id of the input node.
                     * @param node The node to predict the value for.
                     *
                     * @return A pair containing the predicted value, and its MSE w.r.t. the other nodes in the same bucket.
                     */
                    std::pair<double, double> predict(size_t id, const TreeNode & node);

                private:
                    /**
                     * @brief This struct contains the data for each bin.
                     */
                    struct Bin {
                        double avg;
                        double error;
                        unsigned count;
                    };

                    /**
                     * @brief This function updates the bin of the node with its data, and returns the bin.
                     *
                     * @param id The unique id of the input node.
                     * @param node The node to predict the value for.
                     *
                     * @return A reference to the bin that contains the input node.
                     */
                    const Bin & update(size_t id, const TreeNode & node);

                    const MDP::QFunction & ubQ_;
                    size_t entropyBins_, UBBins_;
                    double entropyStep_, UBMin_, UBStep_;

                    //                 id          is_initialized, ei,     ubi,    lb,    err
                    std::unordered_map<size_t, std::tuple<bool,    size_t, size_t, double, double>> nodes_;
                    //              entropy x ub
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

        AI_LOGGER(AI_SEVERITY_DEBUG, "Running SARSOP; POMDP S: " << pomdp.getS() << "; A: " << pomdp.getA() << "; O: " << pomdp.getO());
        AI_LOGGER(AI_SEVERITY_DEBUG, "Initial Belief: " << initialBelief.transpose());

        // ##############################
        // ### Resetting general data ###
        // ##############################

        // Reset delta to the initial parameter set.
        delta_ = initialDelta_;

        // Cache immediate rewards if we can't read the reward function directly.
        if constexpr (!MDP::is_model_eigen_v<M>)
            immediateRewards_ = computeImmediateRewards(pomdp);

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

        // This we use to estimate the UB buckets for each belief. We use more
        // than one since they do the same in the original code.
        const auto initialUbQ = ubQ;

        constexpr unsigned numBins = 2;
        constexpr unsigned entropyBins = 5;
        constexpr unsigned ubBins = 5;
        constexpr unsigned binScaling = 2;

        for (unsigned i = 0; i < numBins; ++i) {
            const unsigned scaling = std::pow(binScaling, i);
            // Each predictor has differently sized buckets.
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

        // ##################
        // ### Begin work ###
        // ##################

        while (true) {
            // Deep sample a branch of the action/observation trees. The
            // sampled nodes (except the last one where we stop) are added to
            // sampledNodes_.
            AI_LOGGER(AI_SEVERITY_DEBUG, "Sampling points...");
            samplePoints(pomdp, lbVList, ubQ, ubV);

            // If we have no nodes it means we stopped at the root, so we have
            // already shrinked the gap enough; we are done.
            if (sampledNodes_.size() == 0) {
                AI_LOGGER(AI_SEVERITY_INFO, "No more points to sample found.");
                break;
            }

            // Backup the nodes we sampled, from (node-before) leaf to root.
            // This updates the lower and upper bounds by adding
            // alphavectors/points to them.
            AI_LOGGER(AI_SEVERITY_DEBUG, "Backing up points...");
            for (auto rIt = std::rbegin(sampledNodes_); rIt != std::rend(sampledNodes_); ++rIt)
                backupNode(*rIt, pomdp, lbVList, ubQ, ubV);

            // # Lower Bound Pruning #

            // We aggressively prune the lbVList based on the beliefs we have
            // explored. This prunes both using direct dominance as well as
            // delta dominance, i.e. vectors count as dominated if they are
            // dominated within a given neighborhood of all their witness
            // beliefs.
            AI_LOGGER(AI_SEVERITY_DEBUG, "Delta pruning...");
            deltaPrune(lbVList);

            // # Upper Bound Pruning #

            // Prune unused beliefs that do not contribute to the upper bound.
            // This means that their value is *higher* than what we can
            // approximate using the other beliefs.
            AI_LOGGER(AI_SEVERITY_DEBUG, "UB pruning...");
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

            AI_LOGGER(AI_SEVERITY_INFO,
                "Root lower bound: " << treeStorage_[0].LB <<
                "; upper bound: " << treeStorage_[0].UB <<
                "; alpha vectors: " << lbVList.size() <<
                "; belief points: " << ubV.first.size());

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
        const double rootGap = (treeStorage_[0].UB - treeStorage_[0].LB) * 0.95;

        int depth = 0;
        double L = treeStorage_[0].LB;
        double U = L + rootGap;

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
            sampledNodes_.push_back(currentNodeId);

            // Precompute this node's children if it was a leaf.
            if (treeStorage_[currentNodeId].children.size() == 0)
                expandLeaf(currentNodeId, pomdp, lbVList, ubQ, ubV);

            // Now we can take a reference as we won't need to allocate again.
            const TreeNode & node = treeStorage_[currentNodeId];

            // Otherwise we keep sampling.
            const auto L1 = std::max(L, node.LB);
            const auto U1 = std::max(U, node.LB + targetGap);

            // TODO: possible do randomization for equally valued actions.
            const auto a1 = node.actionUb;
            // TODO: possible do randomization for equally valued obs.
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

            double Lnorm = 0.0, Unorm = 0.0;
            for (size_t o = 0; o < pomdp.getO(); ++o) {
                if (o == o1) continue;

                const auto & childNode = treeStorage_[node.children[a1][o].id];

                Lnorm += childNode.LB * node.children[a1][o].observationProbability;
                Unorm += childNode.UB * node.children[a1][o].observationProbability;
            }

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
        // Note that we create a pointer as this function will add nodes to
        // treeStorage_, which will possibly re-allocate its storage. This
        // means that references will lose validity. Here we use a pointer to
        // re-assign it when needed.
        TreeNode * nodep = &treeStorage_[id];

        assert(nodep->children.size() == 0);
        // This assert is to say that we shouldn't really be going down a
        // provenly suboptimal path, so this should not really happen.  If it
        // happens, it might be something is broken or I misunderstood
        // something.
        assert(nodep->count > 0);

        // Allocate precompute bound values for future backups
        updateNode(*nodep, pomdp, lbVList, ubQ, ubV, true);

        // Allocate children memory
        nodep->children.resize(boost::extents[pomdp.getA()][pomdp.getO()]);

        for (size_t a = 0; a < pomdp.getA(); ++a) {
            updateBeliefPartial(pomdp, nodep->belief, a, &intermediateBeliefTmp_);

            for (size_t o = 0; o < pomdp.getO(); ++o) {
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

                const auto it = beliefToNode_.find(nextBeliefTmp_);
                if (it != beliefToNode_.end()) {
                    // If the node already existed, we simply point to it, and
                    // increase its reference count.
                    child.id = it->second;
                    if (++treeStorage_[child.id].count == 1) {
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
                child.id = treeStorage_.size();
                beliefToNode_[nextBeliefTmp_] = child.id;

                // Adding a node to treeStorage_ invalidates every single
                // reference we are holding to anything in it, since it may
                // reallocate. Keep it in mind.
                treeStorage_.emplace_back();
                // Re-assign to nodep to get the possibly new pointer.
                nodep = &treeStorage_[id];

                auto & childNode = treeStorage_.back();

                childNode.belief = nextBeliefTmp_;
                childNode.count = 1;
                // Compute UB and LB for this child
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
