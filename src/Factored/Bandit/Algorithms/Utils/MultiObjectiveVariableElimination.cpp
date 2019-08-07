#include <AIToolbox/Factored/Bandit/Algorithms/Utils/MultiObjectiveVariableElimination.hpp>

#include <algorithm>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Prune.hpp>

namespace AIToolbox::Factored::Bandit {
    using MOVE = MultiObjectiveVariableElimination;

    namespace {
        struct Global {
            const Action & A;
            MOVE::Results results;

            size_t agent;
            size_t agentAction;
            MOVE::Factor newFactor;
            MOVE::Factor newCrossSum;
            MOVE::Factor newFactorCrossSum;

            void beginRemoval(size_t agent);
            void initNewFactor();
            void beginCrossSum(size_t agentAction);
            void beginFactorCrossSum();
            void crossSum(const MOVE::Factor & f);
            void endFactorCrossSum();
            void endCrossSum();
            bool isValidNewFactor();
            void mergeFactors(MOVE::Factor & lhs, MOVE::Factor && rhs) const;
            void makeResult(MOVE::GVE::FinalFactors && finalFactors);
        };
    }

    MOVE::Results MOVE::operator()(const Action & A, GVE::Graph & graph) {
        GVE gve;
        Global global{A, {}, 0, 0, {}, {}, {}};

        gve(A, graph, global);

        return global.results;
    }

    /**
     * @brief This function cross-sums the input lists.
     *
     * For each element of tag/value in both inputs, a new value will be
     * returned with a value equal to the element-wise sum of the operands,
     * and tag equal to merged tags of the operands.
     *
     * @param lhs The left hand side.
     * @param rhs The right hand side.
     *
     * @return A new list containing all cross-sums.
     */
    MOVE::Factor crossSumF(const MOVE::Factor & lhs, const MOVE::Factor & rhs) {
        if (!lhs.size()) return rhs;
        if (!rhs.size()) return lhs;
        MOVE::Factor retval;
        retval.reserve(lhs.size() * rhs.size());
        // We do the rhs last since they'll usually be shorter (due to
        // this class usage), so hopefully we can use the cache better.
        for (const auto & lhsVal : lhs) {
            for (const auto & rhsVal : rhs) {
                auto tags = merge(lhsVal.tag, rhsVal.tag);
                auto values = lhsVal.vals + rhsVal.vals;
                // FIXME: C++20, remove useless temporary (they need to fix aggregates).
                retval.emplace_back(MOVE::Entry{std::move(values), std::move(tags)});
            }
        }
        return retval;
    }

    void Global::beginRemoval(size_t currAgent) {
        // We save the currently eliminated agent to initialize the crossSum
        // tag correctly later.
        agent = currAgent;
    }

    void Global::initNewFactor() {
        newFactor.clear();
    }

    void Global::beginCrossSum(size_t currAction) {
        newCrossSum.clear();

        agentAction = currAction;
    }

    void Global::beginFactorCrossSum() {
        // Since we need to compute cross-sums, each factor must be
        // cross-summed separately. This variable is thus the output of
        // cross-summing the next factor with the rules we already have in
        // newCrossSum.
        newFactorCrossSum.clear();
    }

    void Global::crossSum(const MOVE::Factor & factor) {
        // So the idea here is that we are computing results for this
        // particular subset of agents. Here we are working with a single
        // action. However, we may have eliminated agents already. This means
        // that this factor will contain a certain number of rules, which
        // depend on different "already taken" actions of the eliminated
        // agents.
        //
        // During normal VE, we can simply add up all tags since they can't
        // possibly conflict (due to the max operator which always makes us
        // pick the best one). Here instead, the payoffs we cross-sum can't get
        // squashed into a single one and summed, since their tags are no
        // longer guaranteed unique.
        //
        // Thus we get them all, and during the cross/sum we create even more
        // rules, joining their tags together, and possibly merge them if we
        // see equal ones.
        auto tmp = crossSumF(newCrossSum, factor);

        newFactorCrossSum.insert(
            std::end(newFactorCrossSum),
            std::make_move_iterator(std::begin(tmp)),
            std::make_move_iterator(std::end(tmp))
        );
    }

    void Global::endFactorCrossSum() {
        // Here we move back the output of cross-summing the last factor into
        // our actual set of rules, ready to cross-sum the next.
        //
        // Note that in case no factors were found to cross-sum, we don't want
        // to discard our old results, so we only move if we have something!
        if (newFactorCrossSum.size() > 0) {
            // p3 pruning if needed
            newCrossSum = std::move(newFactorCrossSum);
        }
    }

    void Global::endCrossSum() {
        if (newCrossSum.size() > 0) {
            // Add tags
            for (auto & nv : newCrossSum) {
                auto & [first, second] = nv.tag;

                size_t i = 0;
                while (i < first.size() && first[i] < agent) ++i;

                first.insert(std::begin(first) + i, agent);
                second.insert(std::begin(second) + i, agentAction);
            }
            newFactor.insert(
                std::end(newFactor),
                std::make_move_iterator(std::begin(newCrossSum)),
                std::make_move_iterator(std::end(newCrossSum))
            );
        }
    }

    bool Global::isValidNewFactor() {
        // p2.prune(&newFactor);

        return newFactor.size() > 0;
    }

    void Global::mergeFactors(MOVE::Factor & lhs, MOVE::Factor && rhs) const {
        lhs = crossSumF(lhs, rhs);
    }

    void Global::makeResult(MOVE::GVE::FinalFactors && finalFactors) {
        if (finalFactors.size() == 0) return;

        for (const auto & fValue : finalFactors)
            results = crossSumF(results, fValue);

        // P1 pruning
        const auto unwrap = +[](MOVE::Entry & e) -> Rewards & {return e.vals;};

        results.erase(extractDominated(unwrap(results[0]).size(), std::begin(results), std::end(results), unwrap), std::end(results));
    }
}
