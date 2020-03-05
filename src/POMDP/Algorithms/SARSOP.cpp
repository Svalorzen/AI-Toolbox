#include <AIToolbox/POMDP/Algorithms/SARSOP.hpp>

namespace AIToolbox::POMDP {
    SARSOP::SARSOP(double tolerance, double delta) :
            tolerance_(tolerance), initialDelta_(delta) {}

    void addWit(size_t id, VEntry & ve) {
        if (id < (size_t)ve.values.size()) return;
        auto & v = ve.observations;

        // Check if max is already there
        if (sequential_sorted_contains(std::begin(v) + 1, std::begin(v) + v[0], id))
            return;

        // Witness insertion
        const auto it = sequential_sorted_find(std::begin(v) + v[0], std::end(v), id);
        if (it == std::end(v) || *it != id)
            v.insert(it, id);
    }

    void rmWit(size_t id, VEntry & ve) {
        if (id < (size_t)ve.values.size()) return;
        auto & v = ve.observations;

        const auto it = sequential_sorted_find(std::begin(v) + v[0], std::end(v), id);
        if (it != std::end(v) && *it == id)
            v.erase(it);
    }

    void addMax(size_t id, VEntry & ve) {
        // Witness removal
        rmWit(id, ve);
        auto & v = ve.observations;

        // Max insertion
        const auto maxEnd = std::begin(v) + v[0];

        const auto it = sequential_sorted_find(std::begin(v) + 1, maxEnd, id);
        if (it != maxEnd && *it == id)
            return;

        v.insert(it, id);
        ++v[0];
    }

    void rmMax(size_t id, VEntry & ve, bool skipWit = false) {
        auto & v = ve.observations;
        // Max removal
        const auto maxEnd = std::begin(v) + v[0];

        auto it = sequential_sorted_find(std::begin(v) + 1, maxEnd, id);
        if (it == maxEnd || *it != id)
            return;

        v.erase(it);
        --v[0];
        // maxEnd invalid now!

        // Note that corners can't become witnesses.
        if (skipWit || id < (size_t)ve.values.size()) return;

        // Witness insertion
        it = sequential_sorted_find(std::begin(v) + v[0], std::end(v), id);
        if (it == std::end(v) || *it != id)
            v.insert(it, id);
    }

    void SARSOP::updateSubOptimalPaths(TreeNode & root) {
        // Don't go down leaf nodes.
        if (!root.actionData.size())
            return;

        for (auto a = 0; a < root.actionData.cols(); ++a) {
            // Ignore already pruned branches
            if (root.actionData(2, a))
                continue;
            // If this action's upper bound is lower than the node's lower
            // bound, then it is suboptimal, so we remove it.
            if (root.actionData(1, a) < root.LB) {
                // Mark it suboptimal.
                root.actionData(2, a) = true;
                for (size_t o = 0; o < root.children.shape()[1]; ++o) {
                    // Skip impossible children
                    if (root.children[a][o].observationProbability == 0.0)
                        continue;

                    auto & child = treeStorage_[root.children[a][o].id];
                    // Reduce the count of all children, as we are effectively
                    // cutting the edge between this node and theirs.
                    // If no branch is leading to this node anymore, then its
                    // children are also severed from the tree, and we need to
                    // reduce their counts accordingly.
                    assert(child.count); // Sanity check just in case
                    if (--child.count == 0)
                        treePrune(child);
                }
            }
        }
    }

    void SARSOP::deltaPrune(VList & lbVList) {
        // ** IMPORTANT **
        // Look at the comments in the SARSOP header file, where the VList is
        // first initialized, to understand how we use the observations vector
        // to store max/witness points.
        const size_t S = treeStorage_[0].belief.size();

        // Update all reachability counts in the belief tree, so we don't have
        // to check against not useful beliefs. Note that as reachability only
        // depends on a node's LB and UB bounds, and we have only updated those
        // in the sampled nodes, we only need to check them.
        // *Warning*: Keep in mind that sampledNodes_ might have the same id
        // repeated more than once, so the code must handle that.
        // *Warning*: Note that our tree is not actually a tree, but a graph.
        // I'm not sure whether it's possible, but if we end up cutting a link
        // and leaving a "cycle" alive, we are not currently equipped to notice
        // that. Hopefully it won't really change much.
        for (auto id : sampledNodes_)
            updateSubOptimalPaths(treeStorage_[id]);

        // During the backup step we have added new beliefs and alphavectors to
        // lbVList and ubV. Here we remove the elements that are unnecessary
        // (fully dominated) to improve the performance of every other step of
        // the algorithm.
        auto [newBegin, end, oldRmEnd] = extractDominatedIncremental(
            std::begin(lbVList),
            std::end(lbVList) - sampledNodes_.size(),
            std::end(lbVList),
            unwrap
        );
        auto oldEnd = newBegin;
        auto oldRmBegin = end;

        // If no new entries have survived, then the original lbVList has not
        // changed, and thus we have to do nothing else. We just have to clean
        // the removed entries.
        if (newBegin == end) {
            lbVList.erase(end, std::end(lbVList));
            return;
        }

        // First, we need to figure out who is the max vector for the new
        // hyperplanes. We could just run over everybody, but we only check
        // against the old (still good) vectors to avoid repeating a bit of
        // work later. If they are indeed the max in their witness points, it
        // will come back later in the function.
        if (std::begin(lbVList) != oldEnd) {
            // It might happen that one of the new vectors only applied to
            // beliefs which have been proven subOptimal. If that's the case we
            // eliminate these directly here. To do this we simply swap them at
            // the end of the valid range -- this is why we iterate backwards,
            // so we don't have to recheck things.
            auto it = end;
            while (it > newBegin) {
                --it;
                for (size_t i = it->observations[0]; i < it->observations.size(); ++i) {
                    const auto bId = it->observations[i];
                    const auto & node = treeStorage_[bId - S];

                    if (node.count == 0) {
                        rmWit(bId, *it);
                        continue;
                    }

                    const auto & b = node.belief;
                    auto it = findBestAtPoint(b, std::begin(lbVList), oldEnd, nullptr, unwrap);

                    addMax(bId, *it);
                }
                if (it->observations.size() == 1)
                    std::iter_swap(it, --end);
            }
        }

        // Now, we start redistributing the belief points that were held by the
        // dominated alphavectors. These we are sure must go to the new
        // vectors, as:
        //
        // - They can't go to the remaining vectors, or they would already been there.
        // - Since the new vectors fully dominate the removed ones, they
        //   provably have better values so we'll always find something good.
        for (auto it = oldRmBegin; it < oldRmEnd; ++it) {
            // First we distribute the maxes.
            for (size_t i = 1; i < it->observations[0]; ++i) {
                const auto bId = it->observations[i];
                VList::iterator bestIt;
                if (bId < S) {
                    bestIt = findBestAtSimplexCorner(bId, newBegin, end, nullptr, unwrap);
                } else {
                    // Check that the belief is good. Otherwise just ignore it.
                    const auto & node = treeStorage_[bId - S];
                    if (node.count == 0) continue;

                    const auto & b = node.belief;
                    bestIt = findBestAtPoint(b, newBegin, end, nullptr, unwrap);
                }
                addMax(bId, *bestIt);
            }
            // Now we distribute the witness points
            for (size_t i = it->observations[0]; i < it->observations.size(); ++i) {
                const auto bId = it->observations[i];
                const auto & node = treeStorage_[bId - S];

                if (node.count == 0) continue;

                // Here we just find which new vector dominates this one. While
                // there may be more than one, the first we find is fine, as
                // the new vectors can't dominate each other.
                for (auto domIt = newBegin; domIt < end; ++domIt) {
                    if (dominates(domIt->values, it->values)) {
                        addWit(bId, *domIt);
                        break;
                    }
                }
            }
        }

        // Now we have to examine the remaining alphavectors which have not
        // been pruned (they are not directly dominated by the new entries).
        // We still have to check, for every point they are max/witness to,
        // whether the new entries can take their place.
        // If an old alphavector gets all its points stolen from it, it gets
        // pruned.
        for (auto it = std::begin(lbVList); it < oldEnd; ++it) {
            // First we check the maxes. Note that, for a vector still inside
            // our VList, if it gets its max removed it still gets to try to
            // "defend" it as a witness afterwards.
            for (size_t i = 1; i < it->observations[0]; ++i) {
                bool lostMax = false;

                const auto bId = it->observations[i];
                VList::iterator bestIt;
                if (bId < S) {
                    bestIt = findBestAtSimplexCorner(bId, newBegin, end, nullptr, unwrap);

                    if (it->values[bId] < bestIt->values[bId])
                        lostMax = true;
                } else {
                    // Check that the belief is good. Otherwise just ignore it.
                    const auto & node = treeStorage_[bId - S];
                    if (node.count == 0) {
                        rmMax(bId, *it, true);
                        continue;
                    }
                    const auto & b = node.belief;
                    double bestValue;
                    bestIt = findBestAtPoint(b, newBegin, end, &bestValue, unwrap);

                    if (b.transpose() * it->values < bestValue)
                        lostMax = true;
                }

                if (lostMax) {
                    rmMax(bId, *it);
                    addMax(bId, *bestIt);
                }
            }
            // Now we check the witness points. For these, we check using the
            // delta pruning formula that is in the original code.
            for (size_t i = it->observations[0]; i < it->observations.size(); ++i) {
                const auto bId = it->observations[i];
                const auto & node = treeStorage_[bId - S];

                // If the belief is not in use anymore, we simply remove it as
                // a witness point.
                if (node.count == 0) {
                    rmWit(bId, *it);
                    continue;
                }

                const auto & b = node.belief;
                const auto domIt = findBestDeltaDominated(b, it->values, delta_, newBegin, end, unwrap);

                if (domIt != end) {
                    rmWit(bId, *it);
                    addWit(bId, *domIt);
                }
            }
            // If we removed all points from this alphavector (the 1 remaining
            // is the value marking the division between maxes and witnesses,
            // not an actual point), then we mark this vector for removal.
            // We don't remove it yet to be slightly more performant.
            if (it->observations.size() == 1)
                it->action = std::numeric_limits<size_t>::max();
        }

        // Finally, we need to do the delta domination test on the new vectors;
        // it is possible that they may be dominated in their witness points
        // after all.
        for (auto it = newBegin; it < end; ++it) {
            for (size_t i = it->observations[0]; i < it->observations.size(); ++i) {
                const auto bId = it->observations[i];
                const auto & node = treeStorage_[bId - S];

                // Note that these nodes can't be suboptimal, since we have
                // pruned those before.

                const auto & b = node.belief;
                const auto domIt = findBestDeltaDominated(b, it->values, delta_, std::begin(lbVList), end, unwrap);

                if (it != domIt) {
                    rmWit(bId, *it);
                    addWit(bId, *domIt);
                }
            }
            // Again, mark for pruning if that results necessary.
            if (it->observations.size() == 1)
                it->action = std::numeric_limits<size_t>::max();
        }

        // We now remove all marked planes.
        end = std::remove_if(std::begin(lbVList), end, [](const auto & entry) {
                return entry.action == std::numeric_limits<size_t>::max();
        });

        // And finally erase all them from the main list.
        lbVList.erase(end, std::end(lbVList));

        // Update the delta used to do pruning based on the contents of the new
        // lower bound.
        deltaUpdate(lbVList);
    }

    void SARSOP::deltaUpdate(const VList & lbVList) {
        // The original code uses a different heuristic here, but in the end I
        // don't think it changes things that much, and this one is close
        // enough.
        //
        // The idea is to decide we are not pruning enough if we have too many
        // alphavectors which are not the max at any point, but only have delta
        // witnesses. If that's the case, we decrease the delta to make it
        // easier to remove them.
        //
        // On the other hand, if we mostly have alphavectors covering maxes and
        // not too many with only witnesses, we are probably too harsh, so we
        // increase the delta.
        double vectorWithoutMax = 0.0;
        for (const auto & ve : lbVList)
            if (ve.observations[0] == 1)
                ++vectorWithoutMax;

        // How many vectors as a percentage, so we can have thresholds
        // independent of the overall LB size.
        vectorWithoutMax /= lbVList.size();

        constexpr double overpruningThreshold = 0.05;
        constexpr double underpruningThreshold = 0.40;

        // OVERPRUNING: Make it harder to delta prune
        if (vectorWithoutMax < overpruningThreshold) delta_ = std::min(2.0, delta_ * 2.0);
        // UNDERPRUNING Make it easier to delta prune
        if (vectorWithoutMax > underpruningThreshold) delta_ /= 2;
    }

    void SARSOP::treePrune(TreeNode & root) {
        // Don't go down leaf nodes.
        if (!root.actionData.size())
            return;

        for (size_t a = 0; a < root.children.shape()[0]; ++a) {
            // Ignore already pruned branches
            if (root.actionData(2, a))
                continue;

            for (size_t o = 0; o < root.children.shape()[1]; ++o) {
                // Skip impossible children
                if (root.children[a][o].observationProbability == 0.0)
                    continue;

                auto & child = treeStorage_[root.children[a][o].id];

                if (--child.count == 0)
                    treePrune(child);
            }
        }
    }

    void SARSOP::treeRevive(TreeNode & root) {
        // Don't go down leaf nodes.
        if (!root.actionData.size())
            return;

        for (size_t a = 0; a < root.children.shape()[0]; ++a) {
            // Ignore already pruned branches
            if (root.actionData(2, a))
                continue;

            for (size_t o = 0; o < root.children.shape()[1]; ++o) {
                // Skip impossible children
                if (root.children[a][o].observationProbability == 0.0)
                    continue;

                auto & child = treeStorage_[root.children[a][o].id];

                if (++child.count == 1)
                    treeRevive(child);
            }
        }
    }

    // ###############################
    // ### BELIEF VALUE PREDICTION ###
    // ###############################

    double SARSOP::predictValue(size_t id, const TreeNode & node) {
        double retval, error = std::numeric_limits<double>::max();
        for (auto & bin : predictors_) {
            auto [avg, err] = bin.predict(id, node);
            if (err < error) {
                error = err;
                retval = avg;
            }
        }

        retval = std::min(retval, node.UB);
        retval = std::max(retval, node.LB);

        return retval;
    }

    SARSOP::LBPredictor::LBPredictor(size_t entropyBins, size_t UBBins, const MDP::QFunction & ubQ) :
            ubQ_(ubQ), entropyBins_(entropyBins), UBBins_(UBBins), bins_(boost::extents[entropyBins_][UBBins_])
    {
        const double maxEntropy = std::log2(1.0 / ubQ_.rows());
        entropyStep_ = maxEntropy / entropyBins_;

        const Vector cornerVals = ubQ_.rowwise().maxCoeff();

        UBMin_ = cornerVals.minCoeff();
        UBStep_ = (cornerVals.maxCoeff() - UBMin_) / UBBins_;
    }

    std::pair<double, double> SARSOP::LBPredictor::predict(size_t id, const TreeNode & node) {
        const auto & bin = update(id, node);

        if (bin.count == 1)
            return {node.UB, 0.0};

        return {bin.avg, bin.error};
    }

    const SARSOP::LBPredictor::Bin & SARSOP::LBPredictor::update(size_t id, const TreeNode & node) {
        auto & [inBins, ei, ubi, lb, err] = nodes_[id];

        // If we have not done this yet, compute the bucket for this node.
        if (!inBins) {
            const double entropy = getEntropyBase2(node.belief);
            const double ub = (node.belief.transpose() * ubQ_).maxCoeff();

            // Sanity check index bounding
            ei = std::min((size_t)(entropy / entropyStep_), entropyBins_ - 1);
            // We here cast to int to allow for negatives, which get bound to 0
            const auto ubiMax = std::max((int)((ub - UBMin_) / UBStep_), 0);
            ubi = std::min((size_t)ubiMax, UBBins_ - 1);
        }

        auto & bin = bins_[ei][ubi];
        const unsigned afterUpdateCount = bin.count + (!inBins);
        inBins = true;

        bin.avg = (bin.avg * bin.count + node.LB - lb) / afterUpdateCount;
        lb = node.LB;

        // If only a single Belief is in a bin, we compute the error against
        // the node's UB as that's supposed to be the initial target.
        const auto newErr = afterUpdateCount == 1 ?
            (node.UB - lb) * (node.UB - lb):
            (bin.avg - lb) * (bin.avg - lb);

        // Note that the original code does not use MSE, but simply the sum of
        // the squared errors. However, as doing that would basically restrict
        // the choice to bins with fewer beliefs, we normalize the error here.
        bin.error = (bin.error * bin.count + newErr - err) / afterUpdateCount;
        err = newErr;

        bin.count = afterUpdateCount;

        return bin;
    }

    void SARSOP::setTolerance(double tolerance) { tolerance_ = tolerance; }
    double SARSOP::getTolerance() const { return tolerance_; }
    void SARSOP::setDelta(double delta) { initialDelta_ = delta; }
    double SARSOP::getDelta() const { return initialDelta_; }
}
