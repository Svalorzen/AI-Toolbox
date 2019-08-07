#include <AIToolbox/Factored/Bandit/Algorithms/Utils/UCVE.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/Utils/Prune.hpp>
#include <AIToolbox/Factored/Utils/Core.hpp>
#include <AIToolbox/Impl/Logging.hpp>

namespace AIToolbox::Factored::Bandit {
    namespace {
        struct Global {
            using GVE = UCVE::GVE;

            const Action & A;
            const double logtA12;
            double x_u, x_l;

            UCVE::Result result;

            size_t agent;
            size_t agentAction;
            UCVE::Factor newFactor;
            UCVE::Factor newCrossSum;
            UCVE::Factor newFactorCrossSum;

            void beginRemoval(const GVE::Graph &, const GVE::Graph::FactorItList &, size_t);
            void initNewFactor();
            void beginCrossSum(size_t agentAction);
            void beginFactorCrossSum();
            void crossSum(const UCVE::Factor & f);
            void endFactorCrossSum();
            void endCrossSum();
            bool isValidNewFactor();
            void mergeFactors(UCVE::Factor & lhs, UCVE::Factor && rhs) const;
            void makeResult(UCVE::GVE::FinalFactors && finalFactors);
        };
    }

    UCVE::Result UCVE::operator()(const Action & A, const double logtA, GVE::Graph & graph) {
        GVE gve;
        Global global{A, logtA * 0.5, 0.0, 0.0, {}, 0, 0, {}, {}, {}};

        gve(A, graph, global);

        return global.result;
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
    UCVE::Factor crossSumF(const UCVE::Factor & lhs, const UCVE::Factor & rhs) {
        if (!lhs.size()) return rhs;
        if (!rhs.size()) return lhs;
        UCVE::Factor retval;
        retval.reserve(lhs.size() + rhs.size());
        // We do the rhs last since they'll usually be shorter (due to
        // this class usage), so hopefully we can use the cache better.
        for (const auto & lhsVal : lhs) {
            for (const auto & rhsVal : rhs) {
                auto tags = merge(lhsVal.tag, rhsVal.tag);
                auto values = lhsVal.v + rhsVal.v;
                // FIXME: C++20, remove useless temporary (they need to fix aggregates).
                retval.emplace_back(UCVE::Entry{std::move(values), std::move(tags)});
            }
        }
        return retval;
    }

    double computeValue(const UCVE::Entry & e, const double x, const double logtA12) {
        return e.v[0] + std::sqrt((e.v[1] + x) * logtA12);
    };

    void Global::beginRemoval(const GVE::Graph & graph, const GVE::Graph::FactorItList & factors, size_t currAgent) {
        agent = currAgent;
        x_u = x_l = 0.0;
        // We use these iterators to skip the factors for this agent.
        auto skipIt = factors.cbegin(); const auto factorsEnd = factors.cend();
        for (auto it = graph.cbegin(); it != graph.cend(); ++it) {
            // We skip the ones for this agent. Both lists are in the same
            // order so we can keep track of the last duplicate we found to
            // do less work later.
            if (skipIt != factorsEnd && *skipIt == it) {
                ++skipIt;
                continue;
            }
            double currMax = std::numeric_limits<double>::lowest();
            double currMin = std::numeric_limits<double>::max();
            for (const auto & rule : it->getData()) {
                for (const auto & entry : std::get<1>(rule)) {
                    currMax = std::max(currMax, entry.v[1]);
                    currMin = std::min(currMin, entry.v[1]);
                }
            }
            x_u += currMax;
            x_l += currMin;
        }
        AI_LOGGER(AI_SEVERITY_DEBUG, "Current bounds: lower = " << x_l << "; higher = " << x_u);
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

    void Global::crossSum(const UCVE::Factor & factor) {
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
            // We only use prune if we cross-summed more than a
            // single-factor (otherwise it's impossible to have introduced
            // dominated vectors)
            if (newFactorCrossSum.size() > newCrossSum.size() && newFactorCrossSum.size() > 1) {
                // We first eliminate all dominated vectors, then we remove all that
                // can't possibly be useful using the bounds we have computed.
                const auto unwrap = +[](UCVE::Entry & entry) -> UCVE::V & {return entry.v;};

                const auto begin = std::begin(newFactorCrossSum);
                const auto end = std::end(newFactorCrossSum);

                auto bound = extractDominated(2, begin, end, unwrap);

                auto [maxIt, max] = max_element_unary(begin, bound, [x_l = x_l, logtA12 = logtA12](const auto & v) { return computeValue(v, x_l, logtA12); });

                // Put the best first so we can use <= for the pruning (otherwise if we
                // didn't know we would be forced to use < to avoid removing the best)
                std::iter_swap(begin, maxIt);
                newFactorCrossSum.erase(
                    std::remove_if(begin + 1, bound, [max, x_u = x_u, logtA12 = logtA12](const UCVE::Entry & e) { return computeValue(e, x_u, logtA12) <= max; }),
                    end
                );
            }

            newCrossSum = std::move(newFactorCrossSum);
        }
    }

    void Global::endCrossSum() {
        if (newCrossSum.size() > 0) {
            AI_LOGGER(AI_SEVERITY_DEBUG, "Adding entries...");
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
        return newFactor.size() > 0;
    }

    void Global::mergeFactors(UCVE::Factor & lhs, UCVE::Factor && rhs) const {
        lhs = crossSumF(lhs, rhs);
    }

    void Global::makeResult(UCVE::GVE::FinalFactors && finalFactors) {
        AI_LOGGER(AI_SEVERITY_DEBUG, "Picking best final factors...");

        auto & [action, value] = result;
        action.resize(A.size());
        value.setZero();
        for (const auto & fValue : finalFactors) {
            auto & [maxV, maxA] = *max_element_unary(
                std::begin(fValue),
                std::end(fValue),
                [logtA12 = logtA12](const auto & v) { return computeValue(v, 0.0, logtA12); }
            ).first;

            value += maxV;

            // Add tags together
            for (size_t i = 0; i < maxA.first.size(); ++i)
                action[maxA.first[i]] = maxA.second[i];
        }
    }
}
